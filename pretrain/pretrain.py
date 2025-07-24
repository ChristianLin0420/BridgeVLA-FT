"""
BridgeVLA Pre-training Module

This module implements pre-training functionality for RoboPoint detection using PaliGemma,
a vision-language model fine-tuned for robotic pointing tasks. The module includes:

- Dataset handling for robotic detection tasks
- Custom model architecture based on PaliGemma
- Training pipeline with visualization capabilities
- Inference and evaluation utilities

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
"""

# Standard library imports
import argparse
import ast
import datetime
import json
import os
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from einops import rearrange
from PIL import Image, ImageDraw
from safetensors import safe_open
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoProcessor, 
    PaliGemmaForConditionalGeneration, 
    Trainer, 
    TrainingArguments
)

# Local imports
import bridgevla.mvt.utils as mvt_utils
from bridgevla.mvt.raft_utils import ConvexUpSample


# ============================================================================
# Utility Functions
# ============================================================================

def masked_softmax(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Perform independent softmax calculation on non-zero regions of each sample.
    
    This function applies softmax only to the non-zero regions of each heatmap,
    ensuring that the sum of probabilities in non-zero regions equals 1.
    
    Args:
        heatmap: A floating-point tensor with shape (batch_size, height, width)
        eps: Numerical stability coefficient to prevent division by zero
    
    Returns:
        Processed heatmap where the sum of non-zero regions is 1
        
    Example:
        >>> heatmap = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 0.0, 1.0]]])
        >>> result = masked_softmax(heatmap)
        >>> # Non-zero regions sum to 1.0
    """
    # Create mask for non-zero elements
    mask = (heatmap != 0).float()
    
    # Apply mask to input for numerical stability
    stable_input = heatmap * mask 

    # Compute exponential values only for non-zero regions
    exp_vals = torch.exp(stable_input) * mask
    
    # Sum exponentials across spatial dimensions
    sum_exp = exp_vals.sum(dim=(1, 2), keepdim=True)
    
    # Compute normalized softmax with numerical stability
    soft_heatmap = exp_vals / (sum_exp + eps)
    
    return soft_heatmap


def is_list_string(input_string: str) -> bool:
    """
    Check if a string represents a valid Python list when parsed.
    
    Args:
        input_string: String to validate as a list representation
        
    Returns:
        True if the string represents a valid list, False otherwise
        
    Example:
        >>> is_list_string("[1, 2, 3]")
        True
        >>> is_list_string("not a list")
        False
    """
    input_string = input_string.strip()
    
    # Basic format check
    if not (input_string.startswith('[') and input_string.endswith(']') and len(input_string) >= 2):
        return False
    
    try:
        parsed = ast.literal_eval(input_string)
        return isinstance(parsed, list)
    except (SyntaxError, ValueError):
        return False


def convert_xyxy_to_cxcywh(bbox: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    
    Args:
        bbox: Normalized bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Tuple containing (center_x, center_y, width, height) in normalized coordinates
        
    Example:
        >>> bbox = [0.1, 0.2, 0.5, 0.8]
        >>> convert_xyxy_to_cxcywh(bbox)
        (0.3, 0.5, 0.4, 0.6)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return (center_x, center_y, width, height)


def masked_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform position-wise weighted average on a tensor.
    
    Computes the mean across the batch dimension for each spatial position,
    weighted by the number of non-zero elements at that position.
    
    Args:
        tensor: Input tensor of shape (batch_size, width, height) with values in [0, 1]
        
    Returns:
        Weighted average tensor of shape (1, width, height)
        
    Note:
        Positions with all zero values across the batch will remain zero.
    """
    # Calculate mask of non-zero elements
    mask = (tensor != 0).float()
    
    # Count non-zero elements at each position
    count = mask.sum(dim=0, keepdim=True)
    
    # Prevent division by zero
    count = torch.where(count == 0, torch.ones_like(count), count)
    
    # Calculate weighted average
    weighted_sum = tensor.sum(dim=0, keepdim=True) / count

    return weighted_sum


def visualize_points_and_heatmap(
    image: Image.Image, 
    points: List[Tuple[float, float]], 
    heatmap: torch.Tensor, 
    save_path: str,
    point_radius: int = 3
) -> None:
    """
    Visualize an image with point annotations and overlay a corresponding heatmap.
    
    Creates a side-by-side visualization showing the original image with annotated points
    and the same image with a heatmap overlay for visual analysis.
    
    Args:
        image: The original PIL Image object
        points: List of normalized coordinate points in format [(x, y), ...]
                where x, y are in range [0, 1]
        heatmap: Heatmap tensor with shape (1, 224, 224) or (224, 224)
        save_path: File path where the visualization will be saved
        point_radius: Radius in pixels for drawing annotation points
        
    Example:
        >>> points = [(0.5, 0.3), (0.7, 0.8)]
        >>> visualize_points_and_heatmap(image, points, heatmap, "output.png")
    """
    img_width, img_height = image.size
    
    # Scale normalized points to image dimensions
    scaled_points = [(x * img_width, y * img_height) for (x, y) in points]
    
    # Create annotated image
    drawable_image = image.copy()
    draw = ImageDraw.Draw(drawable_image)
    
    # Draw each point as a green circle
    for (x, y) in scaled_points:
        bbox = [
            (x - point_radius, y - point_radius),
            (x + point_radius, y + point_radius)
        ]
        draw.ellipse(bbox, fill='green', outline='green')
    
    # Process heatmap for visualization
    heatmap = heatmap.squeeze()
    # Scale heatmap for better visibility (multiply by 1000)
    heatmap_visual = (heatmap * 255 * 1000).cpu().numpy().astype(np.uint8)
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Annotated image
    ax1.imshow(drawable_image)
    ax1.set_title('Annotated Image')
    ax1.axis('off')
    
    # Right: Heatmap overlay
    ax2.imshow(drawable_image, alpha=0.1)  # Faded background
    ax2.imshow(heatmap_visual, alpha=0.9, cmap='hot')
    ax2.set_title('Heatmap Visualization')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_bboxes_and_heatmap(
    image: Image.Image, 
    bboxes_norm: List[Tuple[float, float, float, float]], 
    heatmap_tensor: torch.Tensor, 
    save_path: str,
    bbox_colors: List[str] = ['red', 'lime', 'cyan', 'yellow'],
    bbox_width: int = 2
) -> None:
    """
    Visualize an image with bounding boxes and overlay a corresponding heatmap.
    
    Creates a side-by-side visualization showing the image with drawn bounding boxes
    and the same image with a heatmap overlay for spatial analysis.
    
    Args:
        image: The original PIL Image object
        bboxes_norm: List of normalized bounding boxes in format [(cx, cy, w, h), ...]
                    where all values are in range [0, 1]
        heatmap_tensor: Heatmap tensor with shape (1, 224, 224) or (224, 224)
        save_path: File path where the visualization will be saved
        bbox_colors: List of color names for cycling through bounding box colors
        bbox_width: Line width in pixels for drawing bounding boxes
        
    Example:
        >>> bboxes = [(0.5, 0.5, 0.3, 0.4), (0.2, 0.8, 0.15, 0.2)]
        >>> visualize_bboxes_and_heatmap(image, bboxes, heatmap, "output.png")
    """
    # Resize image to standard size for consistent visualization
    resized_img = image.resize((224, 224))
    draw = ImageDraw.Draw(resized_img)
    
    # Create color cycle iterator for multiple bounding boxes
    color_cycle = cycle(bbox_colors)
    
    # Draw all bounding boxes
    for bbox in bboxes_norm:
        # Convert normalized center-width-height to pixel coordinates
        cx, cy, w, h = bbox
        x0 = max(0, int((cx - w/2) * 224))
        y0 = max(0, int((cy - h/2) * 224))
        x1 = min(223, int((cx + w/2) * 224))
        y1 = min(223, int((cy + h/2) * 224))
        
        # Get next color from cycle
        current_color = next(color_cycle)
        
        # Draw bounding box rectangle
        draw.rectangle([x0, y0, x1, y1], 
                      outline=current_color, 
                      width=bbox_width)

    # Process heatmap for visualization
    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left: Image with bounding boxes
    ax1.imshow(resized_img)
    ax1.set_title(f'Image with {len(bboxes_norm)} BBoxes')
    ax1.axis('off')
    
    # Right: Heatmap overlay with faded image background
    ax2.imshow(heatmap, cmap='viridis', alpha=0.95)
    ax2.imshow(resized_img, alpha=0.05)
    ax2.set_title('Heatmap Overlay')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_all_params(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load all model parameters from a safetensors checkpoint directory.
    
    This function handles sharded checkpoints by reading the index file
    and loading parameters from all shard files.
    
    Args:
        checkpoint_dir: Path to directory containing safetensors checkpoint files
        
    Returns:
        Dictionary mapping parameter names to tensors (with "module." prefix removed)
        
    Raises:
        FileNotFoundError: If index file or shard files are missing
        json.JSONDecodeError: If index file is malformed
    """
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    
    # Load the index file
    with open(index_path) as f:
        index = json.load(f)
    
    # Merge all sharded parameters
    all_params = {}
    for shard_file in set(index["weight_map"].values()):
        shard_path = os.path.join(checkpoint_dir, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                # Remove the "module." prefix for compatibility
                clean_key = key.replace("module.", "")
                all_params[clean_key] = f.get_tensor(key)
    
    return all_params


# ============================================================================
# Dataset Classes
# ============================================================================

class RoboPointDataset(Dataset):
    """
    Dataset for RoboPoint detection tasks with supervised fine-tuning support.
    
    This dataset handles robotic pointing and detection data, processing conversation
    formats with image-text pairs for training vision-language models on spatial
    understanding tasks.
    
    Attributes:
        samples: List of processed data samples
        image_folder: Path to directory containing images
        res: Resolution parameter for image processing
    """

    def __init__(
        self, 
        image_folder: str, 
        json_detection_path: Optional[str], 
        res: int
    ) -> None:
        """
        Initialize the RoboPointDataset.
        
        Args:
            image_folder: Path to directory containing image files
            json_detection_path: Path to JSON file containing detection annotations
                                If None, no detection samples will be loaded
            res: Target resolution for image processing
        """
        self.samples: List[Dict[str, Any]] = []
        self.image_folder = image_folder
        self.res = res  # Resolution for processor output
        
        # Load detection data if path is provided
        if json_detection_path is not None:
            with open(json_detection_path, "r") as f:
                self.list_data_detection = json.load(f)
            self.samples_detection = self._process_detection_samples(self.list_data_detection)
            self.samples.extend(self.samples_detection)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    

    def _process_detection_samples(self, list_data_dict: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process detection samples from the RoboPoint dataset format.
        
        Parses conversation-style data to extract text prompts and corresponding
        bounding box annotations for detection tasks.
        
        Args:
            list_data_dict: List of conversation data dictionaries
            
        Returns:
            List of processed sample dictionaries with keys:
                - text: Text prompt for the detection task
                - image_path: Full path to the corresponding image
                - raw_label: Raw annotation string (list or coordinates)
                - flag: Detection type identifier ("detection_1" or "detection_2")
                
        Raises:
            AssertionError: If conversation format is invalid or unrecognized
        """
        samples = []
        
        # Template strings for different detection formats
        BBOX_PROMPT_1 = "<image>\nPlease provide the bounding box coordinate of the region this sentence describes: "
        BBOX_PROMPT_1_ALT = "Please provide the bounding box coordinate of the region this sentence describes: "
        BBOX_PROMPT_2 = " Format the result as a list of tuples, i.e. [(x1, y1, w1, h1), (x2, y2, w2, h2), ...], where x and y are the normalized pixel locations of the object centers, and w and h are the normalized object widths and heights. All values of x, y, w, and h should be between 0 and 1."
        
        for source_data in tqdm(list_data_dict, desc="Processing detection samples"):
            conversations = source_data["conversations"]
            num_conversations = len(conversations)
            
            # Ensure conversations come in pairs (question-answer)
            assert num_conversations % 2 == 0, f"Invalid conversation length: {num_conversations}"
            
            # Process each question-answer pair
            for i in range(1, num_conversations, 2):
                answer_value = conversations[i]["value"]
                
                # Only process if answer is a list string (contains coordinates)
                if not is_list_string(answer_value):
                    continue
                    
                question_text = conversations[i-1]["value"]
                
                # Determine detection type and clean text
                if BBOX_PROMPT_1 in question_text or BBOX_PROMPT_1_ALT in question_text:
                    # Type 1: Single bounding box detection
                    cleaned_text = (question_text
                                  .replace(BBOX_PROMPT_1, "")
                                  .replace(BBOX_PROMPT_1_ALT, ""))
                    detection_flag = "detection_1"
                    
                elif BBOX_PROMPT_2 in question_text:
                    # Type 2: Multiple bounding box detection with formatting instructions
                    cleaned_text = question_text.replace(BBOX_PROMPT_2, "")
                    detection_flag = "detection_2"
                    
                else:
                    raise AssertionError(f"Unrecognized detection prompt format: {question_text[:100]}...")
                
                # Create sample dictionary
                sample = {
                    "text": cleaned_text,
                    "image_path": os.path.join(self.image_folder, source_data["image"]),
                    "raw_label": answer_value,
                    "flag": detection_flag
                }
                samples.append(sample)

        return samples

    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - text: Text prompt for the detection task (with <image> tokens preserved)
                - image: PIL Image object in RGB format
                - raw_label: Raw annotation string
                - flag: Detection type identifier
                
        Raises:
            IndexError: If idx is out of range
            FileNotFoundError: If image file doesn't exist
        """
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
            
        sample = self.samples[idx]
        
        # Extract sample components
        text = sample["text"].replace("<image>\n","")
        image_path = sample["image_path"]
        raw_label = sample["raw_label"]
        flag = sample["flag"]
        
        # Load and convert image
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image not found: {image_path}") from e
        
        # Return structured data dictionary
        return {
            "text": text,
            "image": image,
            "raw_label": raw_label,
            "flag": flag
        }


@dataclass
class DataCollator:
    """
    Data collator for batching RoboPoint samples during training.
    
    This collator processes batches of text-image pairs using the PaliGemma processor
    and adds additional metadata needed for training.
    
    Attributes:
        processor: AutoProcessor instance for tokenizing text and processing images
    """
    processor: AutoProcessor 
    
    def __call__(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples into model-ready tensors.
        
        Args:
            data: List of sample dictionaries from RoboPointDataset.__getitem__
            
        Returns:
            Dictionary containing:
                - Tokenized inputs from processor (input_ids, attention_mask, pixel_values)
                - raw_label: List of raw annotation strings
                - flag: List of detection type identifiers
                
        Example:
            >>> batch = collator([sample1, sample2, sample3])
            >>> batch.keys()
            dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'raw_label', 'flag'])
        """
        # Extract batch components
        texts = [sample["text"] for sample in data]
        images = [sample["image"] for sample in data]
        raw_labels = [sample["raw_label"] for sample in data]
        flags = [sample["flag"] for sample in data]
        
        # Process text and images with the model processor
        tokens = self.processor(
            text=texts, 
            images=images,
            return_tensors="pt", 
            padding="longest"
        )
        
        # Add metadata to the batch
        tokens["raw_label"] = raw_labels
        tokens["flag"] = flags

        return tokens
    
def load_dataset(
    processor: AutoProcessor,
    image_folder: str,
    json_detection_path: str,
    res: int
) -> Tuple[RoboPointDataset, DataCollator]:
    """
    Create dataset and data collator for supervised fine-tuning.
    
    Args:
        processor: AutoProcessor instance for text and image processing
        image_folder: Path to directory containing images
        json_detection_path: Path to JSON file with detection annotations
        res: Target resolution for image processing
        
    Returns:
        Tuple of (dataset, data_collator) ready for training
        
    Example:
        >>> processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
        >>> dataset, collator = load_dataset(processor, "images/", "data.json", 224)
    """
    train_dataset = RoboPointDataset(image_folder, json_detection_path, res)
    data_collator = DataCollator(processor=processor)
    return train_dataset, data_collator


# ============================================================================
# Model Classes
# ============================================================================

class RoboPoint_Paligemma(PaliGemmaForConditionalGeneration):
    """
    Custom PaliGemma model for RoboPoint detection tasks.
    
    This model extends PaliGemma with spatial understanding capabilities for robotic
    pointing and detection. It adds an upsampling module to generate heatmaps from
    vision-language features and implements custom loss functions for spatial reasoning.
    
    Attributes:
        vlm_dim: Dimension of vision-language model hidden states
        up0: Upsampling module for generating heatmaps
        num_pat_img: Number of patches per image dimension (224/14 = 16)
        processor: Text and image processor
        _cross_entropy_loss: Loss function for heatmap prediction
    """
    
    def __init__(self, config) -> None:
        """
        Initialize the RoboPoint PaliGemma model.
        
        Args:
            config: Model configuration object containing hidden_size and other parameters
        """
        super().__init__(config)
        
        # Model dimensions and architecture
        self.vlm_dim = config.hidden_size
        self.num_pat_img = 16  # Image patches per dimension (224/14)
        
        # Upsampling module for heatmap generation
        self.up0 = ConvexUpSample(
            in_dim=config.hidden_size,
            out_dim=1,
            up_ratio=14,  # Image patch size
        )
        
        # Processor and loss function
        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
    
    def _extract_image_tokens(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: torch.Tensor,
        batch_size: int,
        num_img: int = 1
    ) -> torch.Tensor:
        """
        Extract image tokens from the hidden states using attention mask.
        
        Args:
            hidden_states: Last layer hidden states from the model
            attention_mask: Attention mask indicating valid tokens
            batch_size: Number of samples in the batch
            num_img: Number of images per sample (currently hardcoded to 1)
            
        Returns:
            Extracted image tokens with shape (batch_size, 256*num_img, vlm_dim)
        """
        image_tokens = []
        
        for i in range(batch_size):
            current_mask = attention_mask[i]
            current_output = hidden_states[i]
            
            # Find valid (non-zero) token positions
            valid_indices = torch.nonzero(current_mask != 0, as_tuple=True)[0]
            valid_tokens = current_output[valid_indices]
            
            # Take the first 256*num_img tokens (image tokens)
            tokens_needed = 256 * num_img
            assert valid_tokens.shape[0] > tokens_needed, \
                f"Not enough tokens: {valid_tokens.shape[0]} < {tokens_needed}"
            
            image_tokens.append(valid_tokens[:tokens_needed])
        
        return torch.stack(image_tokens)
    
    def _generate_target_heatmaps(
        self, 
        raw_labels: List[str], 
        flags: List[str], 
        image_size: int
    ) -> torch.Tensor:
        """
        Generate target heatmaps from raw label annotations.
        
        Args:
            raw_labels: List of raw annotation strings
            flags: List of detection type flags
            image_size: Size of the square image (height/width)
            
        Returns:
            Target heatmaps tensor with shape (batch_size, 1, image_size, image_size)
        """
        heatmaps = []
        
        for raw_label, flag in zip(raw_labels, flags):
            if flag == "detection_1":
                # Single bounding box detection
                coords = ast.literal_eval(raw_label)
                assert isinstance(coords[0], float) and len(coords) == 4, \
                    f"Invalid detection_1 format: {coords}"
                
                bbox = convert_xyxy_to_cxcywh(coords)
                center_point = torch.tensor([[bbox[0], bbox[1]]])
                
                heatmap = mvt_utils.generate_hm_from_pt(
                    center_point.reshape(-1, 2) * image_size,
                    (image_size, image_size),
                    sigma=2,
                    thres_sigma_times=3,
                )
                
            elif flag == "detection_2":
                # Multiple bounding box detection
                coords_list = ast.literal_eval(raw_label)
                assert isinstance(coords_list[0], tuple) and len(coords_list[0]) == 4, \
                    f"Invalid detection_2 format: {coords_list}"
                
                center_points = torch.tensor([[coord[0], coord[1]] for coord in coords_list])
                
                all_heatmaps = mvt_utils.generate_hm_from_pt(
                    center_points.reshape(-1, 2) * image_size,
                    (image_size, image_size),
                    sigma=2,
                    thres_sigma_times=3,
                )
                
                # Fuse multiple heatmaps using masked operations
                heatmap = masked_mean(all_heatmaps)
                heatmap = masked_softmax(heatmap)
                
            else:
                raise ValueError(f"Unknown detection flag: {flag}")
            
            heatmaps.append(heatmap)
        
        return torch.stack(heatmaps)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor, 
        raw_label: List[str], 
        flag: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with spatial loss computation.
        
        Args:
            input_ids: Token IDs with shape (batch_size, seq_len)
            pixel_values: Image tensors with shape (batch_size, 3, 224, 224)
            attention_mask: Attention mask with shape (batch_size, seq_len)
            raw_label: List of raw annotation strings
            flag: List of detection type identifiers
            
        Returns:
            Dictionary containing the computed loss
        """
        # Get model outputs with hidden states
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract dimensions and validate input
        batch_size, _, height, width = pixel_values.shape
        assert height == width, f"Expected square images, got {height}x{width}"
        
        num_img = 1  # Currently hardcoded to 1 image per sample
        
        # Extract image tokens from hidden states
        image_tokens = self._extract_image_tokens(
            outputs.hidden_states[-1], 
            attention_mask, 
            batch_size, 
            num_img
        )
        
        # Reshape tokens to spatial format
        spatial_features = rearrange(
            image_tokens, 
            'b (c h1 h2) w -> b w c h1 h2', 
            c=num_img, 
            h1=self.num_pat_img, 
            h2=self.num_pat_img
        )
        
        # Prepare for upsampling
        upsampling_input = (
            spatial_features.transpose(1, 2)
            .clone()
            .view(batch_size * num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img)
            .to(torch.float32)
        )
        
        # Generate prediction heatmaps
        pred_heatmaps = self.up0(upsampling_input)  # (bs*num_img, 1, 224, 224)
        pred_heatmaps = pred_heatmaps.view(batch_size, num_img, height, width)
        
        # Flatten spatial dimensions for loss computation
        pred_flat = pred_heatmaps.view(batch_size, num_img, height * width).transpose(1, 2)
        
        # Generate target heatmaps
        target_heatmaps = self._generate_target_heatmaps(raw_label, flag, height)
        target_flat = (
            target_heatmaps.view(batch_size, num_img, height * width)
            .transpose(1, 2)
            .to(pred_flat.device)
        )
        
        # Compute loss
        spatial_loss = self._cross_entropy_loss(pred_flat, target_flat).mean()
        
        return {"loss": spatial_loss}  


    def forward_eval(
        self, 
        input_ids: torch.Tensor, 
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor, 
        raw_label: List[str], 
        flag: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for evaluation with additional prediction outputs.
        
        This method is similar to forward() but also returns the prediction heatmaps
        for visualization and analysis purposes.
        
        Args:
            input_ids: Token IDs with shape (batch_size, seq_len)
            pixel_values: Image tensors with shape (batch_size, 3, 224, 224)
            attention_mask: Attention mask with shape (batch_size, seq_len)
            raw_label: List of raw annotation strings (for loss computation)
            flag: List of detection type identifiers (for loss computation)
            
        Returns:
            Dictionary containing:
                - loss: Computed spatial loss
                - q_trans: Raw prediction heatmaps for visualization
        """
        # Get model outputs with hidden states
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract dimensions and validate input
        batch_size, _, height, width = pixel_values.shape
        assert height == width, f"Expected square images, got {height}x{width}"
        
        num_img = 1  # Currently hardcoded to 1 image per sample
        
        # Extract image tokens from hidden states
        image_tokens = self._extract_image_tokens(
            outputs.hidden_states[-1], 
            attention_mask, 
            batch_size, 
            num_img
        )
        
        # Reshape tokens to spatial format
        spatial_features = rearrange(
            image_tokens, 
            'b (c h1 h2) w -> b w c h1 h2', 
            c=num_img, 
            h1=self.num_pat_img, 
            h2=self.num_pat_img
        )
        
        # Prepare for upsampling
        upsampling_input = (
            spatial_features.transpose(1, 2)
            .clone()
            .view(batch_size * num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img)
            .to(torch.float32)
        )
        
        # Generate prediction heatmaps
        pred_heatmaps = self.up0(upsampling_input)  # (bs*num_img, 1, 224, 224)
        pred_heatmaps = pred_heatmaps.view(batch_size, num_img, height, width)
        
        # Flatten spatial dimensions for loss computation
        pred_flat = pred_heatmaps.view(batch_size, num_img, height * width).transpose(1, 2)
        
        # Generate target heatmaps for loss computation
        target_heatmaps = self._generate_target_heatmaps(raw_label, flag, height)
        target_flat = (
            target_heatmaps.view(batch_size, num_img, height * width)
            .transpose(1, 2)
            .to(pred_flat.device)
        )
        
        # Compute loss
        spatial_loss = self._cross_entropy_loss(pred_flat, target_flat).mean()
        
        return {
            "loss": spatial_loss,
            "q_trans": pred_flat  # Raw predictions for visualization
        } 



# ============================================================================
# Training Pipeline Classes
# ============================================================================

class Pretrain_RoboPoint_Palligemma:
    """
    Main training pipeline for RoboPoint PaliGemma model.
    
    This class handles the complete training workflow including:
    - Model initialization and checkpoint loading
    - Training configuration management
    - Inference and evaluation
    - Visualization utilities
    
    Attributes:
        config: Training configuration loaded from YAML file
        model_id: Hugging Face model identifier
        processor: Text and image processor
        device: Compute device (CPU/CUDA)
        pretrained_model: Loaded model for inference (when not training)
    """
    
    def __init__(self, pretrain: bool, config_path: str) -> None:
        """
        Initialize the training pipeline.
        
        Args:
            pretrain: If True, set up for training. If False, load pretrained model for inference
            config_path: Path to YAML configuration file
            
        Raises:
            FileNotFoundError: If config file or checkpoint directory not found
            RuntimeError: If model loading fails
        """
        # Load configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Configuration file not found: {config_path}") from e
            
        # Model configuration
        self.model_id = "google/paligemma-3b-pt-224"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        # Set up for inference if not training
        if not pretrain:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load checkpoint
            checkpoint_dir = self.config["checkpoint_dir"]
            try:
                all_params = load_all_params(checkpoint_dir)
                self.pretrained_model = RoboPoint_Paligemma.from_pretrained(
                    self.model_id, 
                    trust_remote_code=True
                )
                
                # Load state dict with error reporting
                missing_keys, unexpected_keys = self.pretrained_model.load_state_dict(
                    all_params, 
                    strict=False
                )
                
                print(f"Model loaded successfully!")
                print(f"Missing keys: {missing_keys}")  # Should only contain "lm_head"
                print(f"Unexpected keys: {unexpected_keys}")  # Should be empty
                
                # Clean up and move to device
                del all_params
                self.pretrained_model.to(self.device)
                
            except Exception as e:
                raise RuntimeError(f"Failed to load pretrained model: {e}") from e

    def test_inference(
        self, 
        image_folder: str, 
        json_detection_path: str, 
        res: int = 224,
        sample_stride: int = 300
    ) -> None:
        """
        Run inference testing on a subset of the dataset with visualization.
        
        This method evaluates the pretrained model on sample data and generates
        visualizations showing the predicted heatmaps overlaid on the original images.
        
        Args:
            image_folder: Path to directory containing images
            json_detection_path: Path to JSON file with detection annotations
            res: Target resolution for image processing
            sample_stride: Step size for sampling test examples (every Nth sample)
            
        Raises:
            ValueError: If pretrained_model is not loaded
            FileNotFoundError: If save path directory doesn't exist
        """
        if not hasattr(self, 'pretrained_model'):
            raise ValueError("Pretrained model not loaded. Initialize with pretrain=False.")
            
        # Load test dataset
        test_dataset = RoboPointDataset(image_folder, json_detection_path, res)
        save_path = self.config["test_save_path"]
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"Testing on {len(test_dataset)} samples with stride {sample_stride}")
        
        # Sample from dataset in reverse order
        sample_indices = range(len(test_dataset) - 1, 0, -sample_stride)
        
        self.pretrained_model.eval()
        
        for idx, i in enumerate(sample_indices):
            print(f"\nProcessing sample {idx + 1}/{len(sample_indices)} (dataset index {i})")
            
            # Get sample and prepare batch
            sample = test_dataset[i]
            batch_data = {
                "text": [sample["text"]],
                "images": [sample["image"]],
                "flag": [sample["flag"]],
                "raw_label": [sample["raw_label"]]
            }
            
            # Process through model processor
            tokens = self.processor(
                text=batch_data["text"], 
                images=batch_data["images"],
                return_tensors="pt", 
                padding="longest"
            )
            tokens["flag"] = batch_data["flag"]
            tokens["raw_label"] = batch_data["raw_label"]
            tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in tokens.items()}
            
            # Run inference
            with torch.no_grad():
                output_dict = self.pretrained_model.forward_eval(**tokens)
            
            # Extract and process predictions
            loss = output_dict["loss"].item()
            raw_predictions = output_dict["q_trans"].view(224, 224, 1).detach().cpu()
            
            # Convert to normalized heatmap
            pred_flat = raw_predictions.view(-1)
            normalized_heatmap = torch.nn.functional.softmax(pred_flat, dim=0)
            heatmap = normalized_heatmap.view(224, 224, 1)
            
            # Log results
            print(f"Loss: {loss:.4f}")
            print(f"Label: {sample['raw_label']}")
            print(f"Text: {sample['text'][:100]}...")
            print(f"Heatmap sum: {torch.sum(heatmap):.4f}")
            
            # Generate visualization based on detection type
            sample_save_path = save_path.replace('.png', f'_sample_{i}.png')
            
            if sample["flag"] == "detection_1":
                # Single bounding box
                coords = ast.literal_eval(sample["raw_label"])
                bbox = convert_xyxy_to_cxcywh(coords)
                visualize_bboxes_and_heatmap(
                    sample["image"], [bbox], heatmap, sample_save_path,
                    bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                    bbox_width=2
                )
                
            elif sample["flag"] == "detection_2":
                # Multiple bounding boxes
                coords_list = ast.literal_eval(sample["raw_label"])
                visualize_bboxes_and_heatmap(
                    sample["image"], coords_list, heatmap, sample_save_path,
                    bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                    bbox_width=2
                )
                
            else:
                raise ValueError(f"Unknown detection flag: {sample['flag']}")
            
            print(f"Visualization saved to: {sample_save_path}")
        
        print(f"\nInference testing completed! Results saved to {os.path.dirname(save_path)}")          



    def pretrain(self,image_folder,
                json_detection_path,
                res=224,
                freeze_vision_tower=True):
        
        accelerator = Accelerator()
        
        train_ds, collate_fn = load_dataset(self.processor,image_folder,json_detection_path,res)
        model = RoboPoint_Paligemma.from_pretrained(
            self.model_id,
            quantization_config=None,
            device_map=None, 
            torch_dtype=torch.bfloat16, 
        )
        model.up0=model.up0.to(torch.float32)
        model.gradient_checkpointing_enable()  # Can significantly reduce GPU memory consumption

        freeze_names = ["lm_head", "embed_tokens"]
        if freeze_vision_tower:
            freeze_names.append("vision_tower")
        for name, param in model.named_parameters():
            if any(freeze_name in name for freeze_name in freeze_names):
                param.requires_grad_(False)
        from torch.optim import AdamW
        lr=float(self.config["lr"])   #5e-5 
        bs=self.config["bs"]       #48
        optimizer = AdamW(model.parameters(), lr=lr)
        model, optimizer = accelerator.prepare(model, optimizer)  
        current_time = datetime.datetime.now()
        folder_name = current_time.strftime("%Y%m%d_%H%M%S")
        exp_name=self.config["exp_name"]
        output_path=os.path.join(self.config["output_dir"],exp_name,folder_name)

        args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=bs,  # Adjust single-GPU batch size to 8
            gradient_accumulation_steps=self.config["gradient_accumulation_steps"],   # Gradient accumulation steps
            learning_rate=lr,
            warmup_steps=self.config["warmup_steps"],
            optim="adamw_torch_fused",  # Use the fused optimizer
            bf16=True,                  # Enable BF16 mixed precision
            logging_steps=self.config["logging_steps"],
            logging_strategy="steps",  # Explicitly specify logging by steps
            save_strategy="steps",
            save_steps=self.config["save_steps"],
            save_total_limit=self.config["save_total_limit"],
            dataloader_num_workers=self.config["dataloader_num_workers"],         # Increase data loading threads
            dataloader_pin_memory=True,       # Enable memory pinning
            gradient_checkpointing=False,      # Activate gradient checkpointing
            report_to=["wandb"],
            no_cuda=False,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False
        )

        if accelerator.is_main_process:  
            import wandb
            wandb.login(key="")
            wandb.init(
                entity=self.config["wandb_entity"],
                project=self.config["wandb_project"],
                name=exp_name,
                config=vars(args),
            )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=collate_fn,
            optimizers=(optimizer, None))
        trainer.train()


if __name__=="__main__":

    # test dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=int, default=3)
    parser.add_argument("--config_path", type=str, default="pretrain_config.yaml")
    parser.add_argument("--json_detection_path", type=str, default="/mnt/bn/lpy-lq/hugging_face/RoboPoint_Data/detection_data.json")
    parser.add_argument("--image_folder", type=str, default="/mnt/bn/lpy-lq/hugging_face/RoboPoint_Data/images")
    args = parser.parse_args()
    json_detection_path=args.json_detection_path
    image_folder=args.image_folder
    branches=args.branches  # 1: visualization 2: pretrain 3: evaluation
    if branches==1:
        #visualize dataset
        test_dataset=RoboPointDataset(image_folder=image_folder,json_detection_path=json_detection_path,res=224)
        print("Total samples:",len(test_dataset))
        save_path="./debug.png"
        for i in range(0,len(test_dataset),1000):
            data=test_dataset[i]
            text=data["text"]
            image=data["image"]
            raw_label=data["raw_label"]
            flag=data["flag"]
            print("Image Size:",image.size)
            print("Text:",text)
            if flag=="detection_1":
                answer_points=ast.literal_eval(raw_label)
                assert type(answer_points[0]) is float and len(answer_points)==4
                bbox=convert_xyxy_to_cxcywh(answer_points)
                label=torch.tensor([[bbox[0],bbox[1]]])
                action_trans_now = mvt_utils.generate_hm_from_pt(
                label.reshape(-1, 2) * 224 , # hardcode
                (224,224),
                sigma=2, # hardcode
                thres_sigma_times=3,  # hardcode
                )  # check it carefully   
                visualize_bboxes_and_heatmap(image, [bbox], action_trans_now, save_path,
                                            bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                                            bbox_width=2)
            elif flag=="detection_2":
                answer_points=ast.literal_eval(raw_label)
                assert type(answer_points[0]) is tuple and len(answer_points[0])==4
                labels=torch.tensor([ [answer_point[0],answer_point[1]] for answer_point in answer_points ])
                action_trans_all = mvt_utils.generate_hm_from_pt(
                                        labels.reshape(-1, 2)*224,
                                        (224, 224),
                                        sigma=2, # hardcode
                                        thres_sigma_times=3,  # hardcode
                                        )   
                # fuse the action_trans
                action_trans_now=masked_mean(action_trans_all)
                action_trans_now=masked_softmax(action_trans_now)
                visualize_bboxes_and_heatmap(image, answer_points, action_trans_now, save_path,
                                            bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                                            bbox_width=2)            
            else:
                assert False

    elif branches==2:

        # pretrain with detection
        pipeline=Pretrain_RoboPoint_Palligemma(pretrain=True,config_path=args.config_path)

        pipeline.pretrain(image_folder=image_folder,json_detection_path=json_detection_path,res=224,freeze_vision_tower=True)
    elif branches==3:
        # # test the pretrained checkpoints
        pipeline=Pretrain_RoboPoint_Palligemma(pretrain=False,config_path=args.config_path)
        pipeline.test_inference(image_folder=image_folder,json_detection_path=json_detection_path,res=224)
    else:
        assert False






