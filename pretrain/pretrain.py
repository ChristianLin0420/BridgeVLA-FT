'''
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
'''
import torch
import argparse
from PIL import Image
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from torch import nn
import torch.nn as nn
from bridgevla.mvt.raft_utils import ConvexUpSample
import bridgevla.mvt.utils as mvt_utils
from einops import rearrange
import json
import os
from tqdm import tqdm  
from PIL import Image, ImageDraw
import datetime
from safetensors import safe_open
import json
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
import ast
from itertools import cycle
from PIL import Image, ImageDraw
from accelerate import Accelerator
import matplotlib.pyplot as plt
import os
import yaml  


def masked_softmax(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Perform independent softmax calculation on non-zero regions of each sample.
    
    Args:
        heatmap : Tensor - A floating-point tensor with shape (bs, H, W).
        eps     : float - Numerical stability coefficient (default 1e-8).
    
    Returns:
        soft_heatmap : Tensor - Processed heatmap where the sum of non-zero regions is 1.
    """

    mask = (heatmap != 0).float()
    
    stable_input = heatmap * mask 

    exp_vals = torch.exp(stable_input) * mask
    
    sum_exp = exp_vals.sum(dim=(1, 2), keepdim=True)  # 
    
    soft_heatmap = exp_vals / (sum_exp + eps)
    
    return soft_heatmap




def is_list_string(s):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']') and len(s) >= 2):
        return False
    
    try:
        parsed = ast.literal_eval(s)
        return isinstance(parsed, list)
    except (SyntaxError, ValueError):
        return False



def visualize_points_and_heatmap(image, points, heatmap, save_path,point_radius=3):
    """
    Visualize an image with annotations and its corresponding heatmap.
    
    Args:
        image: PIL.Image object - The original image.
        points: list of tuples - A list of normalized coordinate points in the format [(x,y),...].
        heatmap: numpy.ndarray - A heatmap with shape (1,224,224).
        point_radius: int - The radius (in pixels) of the drawn points.
    """
    img_width, img_height = image.size
    scaled_points = [(x * img_width, y * img_height) for (x, y) in points]
    
    drawable_image = image.copy()
    draw = ImageDraw.Draw(drawable_image)
    
    for (x, y) in scaled_points:
        bbox = [
            (x - point_radius, y - point_radius),
            (x + point_radius, y + point_radius)
        ]
        draw.ellipse(bbox, fill='green', outline='green')
    
    heatmap = heatmap.squeeze()  
    heatmap = (heatmap * 255*1000).numpy().astype(np.uint8)  # *1000 for visualization
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(drawable_image)
    ax1.set_title('Annotated Image')
    ax1.axis('off')
    
    ax2.imshow(drawable_image, alpha=0.1)  # reduce the background display
    heatmap_display = ax2.imshow(heatmap, alpha=0.9)
    ax2.set_title('Heatmap Visualization')
    ax2.axis('off')  
    
    plt.savefig(save_path)



def convert_xyxy_to_cxcywh(bbox):
    """
    Convert (x1, y1, x2, y2) to (cx, cy, w, h).
    
    Args:
        bbox (list/tuple): Normalized bounding box coordinates [x1, y1, x2, y2].
    
    Returns:
        tuple: (cx, cy, w, h) in normalized representation.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return (cx, cy, w, h)




def masked_mean(tensor):
    """
    Perform position-wise weighted average on a tensor of shape (bs, W, H).
    The weight of each position is the number of non-zero elements at that position.
    The input value range should be in [0, 1].
    """
    # Calculate the mask of non-zero elements
    mask = (tensor != 0).float()
    
    # Calculate the number of non-zero elements at each position (denominator)
    count = mask.sum(dim=0, keepdim=True)
    
    # Prevent division by zero: Set positions with zero denominator to 1
    count = torch.where(count == 0, torch.ones_like(count), count)
    
    # Calculate the weighted average
    summed = tensor.sum(dim=0, keepdim=True) / count

    return summed


def visualize_bboxes_and_heatmap(image, bboxes_norm, heatmap_tensor, save_path,
                               bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                               bbox_width=2):
    """
    Visualize an image with bounding boxes and overlay a heatmap.
    
    Args:
        image         : PIL.Image    - The original image object.
        bboxes_norm   : list of tuples - A list of normalized bounding boxes [(cx, cy, w, h), ...].
        heatmap_tensor: torch.Tensor  - A heatmap tensor with shape (1, 224, 224).
        bbox_colors   : list          - A cyclic list of bounding box colors.
        bbox_width    : int           - The line width (in pixels) of the bounding boxes.
    """
    # Convert the image to the target size
    resized_img = image.resize((224, 224))
    draw = ImageDraw.Draw(resized_img)
    
    # Create a color cycle iterator
    color_cycle = cycle(bbox_colors)
    
    # Draw all bounding boxes
    for bbox in bboxes_norm:
        # Convert normalized coordinates to actual pixel coordinates
        cx, cy, w, h = bbox
        x0 = max(0, int((cx - w/2) * 224))
        y0 = max(0, int((cy - h/2) * 224))
        x1 = min(223, int((cx + w/2) * 224))
        y1 = min(223, int((cy + h/2) * 224))
        
        # Get the current color
        current_color = next(color_cycle)
        
        # Draw the bounding box
        draw.rectangle([x0, y0, x1, y1], 
                      outline=current_color, 
                      width=bbox_width)

    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the image with bounding boxes
    ax1.imshow(resized_img)
    ax1.set_title(f'Image with {len(bboxes_norm)} BBoxes')
    ax1.axis('off')
    
    # Display the heatmap and overlay a semi-transparent original image
    heatmap_display = ax2.imshow(heatmap, cmap='viridis', alpha=0.95)
    ax2.imshow(resized_img, alpha=0.05)
    ax2.set_title('Heatmap Overlay')
    ax2.axis('off')
    
    plt.savefig(save_path)



class RoboPointDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,image_folder,json_detection_path,res):
        self.samples=[]
        self.image_folder=image_folder
        self.res=res # Here, res refers to the result after processor processing
        if json_detection_path is not None:
            self.list_data_detection = json.load(open(json_detection_path, "r"))
            self.samples_detection=self.initial_samples_detection(self.list_data_detection)
            self.samples.extend(self.samples_detection)

    def __len__(self):
        return len(self.samples)
    

    def initial_samples_detection(self, list_data_dict):
        '''
        process the detection samples in the RoboPoint Dataset
        '''
        samples = []
        for source_data in tqdm(list_data_dict):
            length_conversations=len(source_data["conversations"])
            assert length_conversations % 2 ==0 
            for i in range(1,length_conversations,2):
                if is_list_string(source_data["conversations"][i]["value"]):
                    matched_string1 ="<image>\nPlease provide the bounding box coordinate of the region this sentence describes: "
                    matched_string1_ ="Please provide the bounding box coordinate of the region this sentence describes: "
                    matched_string2 =" Format the result as a list of tuples, i.e. [(x1, y1, w1, h1), (x2, y2, w2, h2), ...], where x and y are the normalized pixel locations of the object centers, and w and h are the normalized object widths and heights. All values of x, y, w, and h should be between 0 and 1."
                    text = source_data["conversations"][i-1]["value"]
                    if matched_string1 in text or matched_string1_ in text:
                        text=text.replace(matched_string1,"")
                        text=text.replace(matched_string1_,"")
                        sample={}
                        sample["text"]=text
                        sample["image_path"]=os.path.join(self.image_folder,source_data["image"])
                        sample["raw_label"]=source_data["conversations"][i]["value"]
                        sample["flag"]="detection_1"
                        samples.append(sample)  
                    elif matched_string2 in text:
                        text=text.replace(matched_string2,"")
                        sample={}
                        sample["text"]=text
                        sample["image_path"]=os.path.join(self.image_folder,source_data["image"])
                        sample["raw_label"]=source_data["conversations"][i]["value"]
                        sample["flag"]="detection_2"
                        samples.append(sample)                                                     
                    else:
                        assert False

                else:
                    continue

        return samples

    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample["text"].replace("<image>\n","")
        image_path = sample["image_path"]
        raw_label = sample["raw_label"]
        flag=sample["flag"]
        image = Image.open(image_path).convert("RGB")
        data_dict={}
        data_dict["text"]=text
        data_dict["image"]=image
        data_dict["raw_label"]=raw_label
        data_dict["flag"]=flag
        return data_dict


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""
    processor: AutoProcessor 
    def __call__(self, data):
        texts = [ex["text"] for ex in data]#[text1,text2,...]
        images = [ex["image"] for ex in data]# bs,2   
        raw_label = [ex["raw_label"] for ex in data]#[[img1],[img2]...]  bs,1  PIL image
        flag= [ex["flag"] for ex in data]
        tokens = self.processor(text=texts, images=images,return_tensors="pt", padding="longest")
        tokens["raw_label"]=raw_label
        tokens["flag"]=flag

        return tokens
    
def load_dataset(processor,image_folder,json_detection_path,res):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = RoboPointDataset(image_folder,json_detection_path,res)
    data_collator = DataCollator(processor=processor)
    return train_dataset,data_collator


class RoboPoint_Paligemma(PaliGemmaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.vlm_dim=config.hidden_size
        self.up0 = ConvexUpSample(
                in_dim=config.hidden_size,
                out_dim=1,
                up_ratio=14,  # hardcode  img_patch_size
            )  
        self.num_pat_img = 16 # hardcode  224/14
        self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, input_ids, pixel_values,attention_mask,raw_label,flag):
        '''
        input_ids: bs,seq_len   
        pixel_values: bs*num_img,3,224,224
        attention_mask: bs,seq_len
        raw_label: [[],[]...] 
        flag: [flag1,flag2,...]  The flag of each sample, indicating which type of detection data it comes from
        '''
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states 

        x = hidden_states[-1] 
        image_tokens= []
        bs, img_feat_dim, h, w = pixel_values.shape 
        assert h==w
        num_img=1 # hardcode
        # Todo Vectorize
        for i in range(bs):
            # Get the ids and output of the current batch
            current_ids =attention_mask[i]
            current_output = x[i]
            
            # Extract tokens corresponding to non-zero ids
            non_zero_indices = torch.nonzero(current_ids != 0, as_tuple=True)[0] # Find the indices of non-zero ids
            non_zero_output = current_output[non_zero_indices]  # Extract the token outputs corresponding to these non-zero ids
            
            # Take the first 256 tokens (if the number of non-zero tokens is greater than 256, take the first 256)
            assert non_zero_output.shape[0] > 256*num_img
            non_zero_output = non_zero_output[:256*num_img]
            
            # Add the processed output to the new output list
            image_tokens.append(non_zero_output)

        # Combine the new outputs into a single tensor
        image_tokens = torch.stack(image_tokens) # bs,256,vlm_dim
        x = rearrange(image_tokens, 'b (c h1 h2) w -> b w c h1 h2', c=num_img, h1=self.num_pat_img, h2=self.num_pat_img) 

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img
            )
        )
        x=x.to(torch.float32) 
        
        trans = self.up0(x) # bs*num_img,1,224,224
        trans = trans.view(bs, num_img, h, w) # bs,num_img,224,224
        assert h==w
        q_trans=trans.view(bs,num_img,h*w).transpose(1,2)# bs,50176,num_img

        # get action_trans
        action_trans=[]
        for i in range(bs):
            flag_now=flag[i]
            raw_label_now=raw_label[i]
            if flag_now=="detection_1":
                answer_points=ast.literal_eval(raw_label_now)
                assert type(answer_points[0]) is float and len(answer_points)==4
                bbox=convert_xyxy_to_cxcywh(answer_points)
                label=torch.tensor([[bbox[0],bbox[1]]])
                action_trans_now = mvt_utils.generate_hm_from_pt(
                label.reshape(-1, 2) * h,
                (w, h),
                sigma=2, # hardcode
                thres_sigma_times=3,  # hardcode
                )
                action_trans.append(action_trans_now)                       
            elif flag_now=="detection_2":
                answer_points=ast.literal_eval(raw_label_now)
                assert type(answer_points[0]) is tuple and len(answer_points[0])==4
                labels=torch.tensor([ [answer_point[0],answer_point[1]] for answer_point in answer_points ])
                action_trans_all = mvt_utils.generate_hm_from_pt(
                                        labels.reshape(-1, 2)*h,
                                        (w, h),
                                        sigma=2, # hardcode
                                        thres_sigma_times=3,  # hardcode
                                        )    
                # fuse the action_trans
                action_trans_now=masked_mean(action_trans_all)
                action_trans_now=masked_softmax(action_trans_now)
                action_trans.append(action_trans_now)
            else:
                assert False

        action_trans=torch.stack(action_trans)
        action_trans = action_trans.view(bs,num_img, h * w).transpose(1, 2).clone().to(q_trans.device)
        trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()

        return {"loss": trans_loss}  


    def forward_eval(self, input_ids, pixel_values,attention_mask,raw_label,flag):
        '''
        input_ids: bs,seq_len   
        pixel_values: bs*num_img,3,224,224
        attention_mask: bs,seq_len
        '''
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states 

        x = hidden_states[-1] 
        image_tokens= []
        bs, img_feat_dim, h, w = pixel_values.shape 
        # Todo Vectorize
        num_img=1 # hardcode
        for i in range(bs):
            current_ids =attention_mask[i]
            current_output = x[i]
            
            non_zero_indices = torch.nonzero(current_ids != 0, as_tuple=True)[0]  
            non_zero_output = current_output[non_zero_indices]  
            
            assert non_zero_output.shape[0] > 256*num_img
            non_zero_output = non_zero_output[:256*num_img]
            
            image_tokens.append(non_zero_output)

        image_tokens = torch.stack(image_tokens) # bs,256,vlm_dim
        x = rearrange(image_tokens, 'b (c h1 h2) w -> b w c h1 h2', c=num_img, h1=self.num_pat_img, h2=self.num_pat_img)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * num_img, self.vlm_dim, self.num_pat_img, self.num_pat_img
            )
        )
        x=x.to(torch.float32) 
        
        trans = self.up0(x) # bs*num_img,1,224,224
        trans = trans.view(bs, num_img, h, w) # bs,num_img,224,224
        assert h==w
        q_trans=trans.view(bs,num_img,h*w).transpose(1,2)# bs,50176,num_img
        # get action_trans
        action_trans=[]
        for i in range(bs):
            flag_now=flag[i]
            raw_label_now=raw_label[i]
            if flag_now=="detection_1":
                answer_points=ast.literal_eval(raw_label_now)
                assert type(answer_points[0]) is float and len(answer_points)==4
                bbox=convert_xyxy_to_cxcywh(answer_points)
                label=torch.tensor([[bbox[0],bbox[1]]])
                action_trans_now = mvt_utils.generate_hm_from_pt(
                label.reshape(-1, 2) * h,
                (w, h),
                sigma=2, # hardcode
                thres_sigma_times=3,  # hardcode
                )  
                action_trans.append(action_trans_now)                       
            elif flag_now=="detection_2":
                answer_points=ast.literal_eval(raw_label_now)
                assert type(answer_points[0]) is tuple and len(answer_points[0])==4
                labels=torch.tensor([ [answer_point[0],answer_point[1]] for answer_point in answer_points ])
                action_trans_all = mvt_utils.generate_hm_from_pt(
                                        labels.reshape(-1, 2)*h,
                                        (w, h),
                                        sigma=2, # hardcode
                                        thres_sigma_times=3,  # hardcode
                                        ) 
                # fuse the action_trans
                action_trans_now=masked_mean(action_trans_all)
                action_trans_now=masked_softmax(action_trans_now)
                action_trans.append(action_trans_now)
            else:
                assert False

        action_trans=torch.stack(action_trans)
        action_trans = action_trans.view(bs,num_img, h * w).transpose(1, 2).clone().to(q_trans.device)
        trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()

        return {"loss": trans_loss,"q_trans":q_trans} 



def load_all_params(checkpoint_dir):
    # Load the index file
    with open(f"{checkpoint_dir}/model.safetensors.index.json") as f:
        index = json.load(f)
    
    # Merge all sharded parameters
    all_params = {}
    for shard_file in set(index["weight_map"].values()):
        with safe_open(f"{checkpoint_dir}/{shard_file}", framework="pt") as f:
            for key in f.keys():
                # Remove the "module." prefix
                clean_key = key.replace("module.", "")
                all_params[clean_key] = f.get_tensor(key)
    return all_params

class Pretrain_RoboPoint_Palligemma:
    def __init__(self,pretrain,config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)        
        self.model_id = "google/paligemma-3b-pt-224"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        if not pretrain:
            self.device="cuda"
            checkpoint_dir=self.config["checkpoint_dir"] 
            all_params=load_all_params(checkpoint_dir)
            self.pretrained_model = RoboPoint_Paligemma.from_pretrained(self.model_id, trust_remote_code=True)
            
            missing_keys, unexpected_keys = self.pretrained_model.load_state_dict(all_params,strict=False)
            print("Missing keys  base:", missing_keys)  # Should only contain "lm_head"
            print("Unexpected keys base:", unexpected_keys)  # Should be an empty list
            del all_params
            self.pretrained_model.to(self.device)

    def test_inference(self,image_folder,json_detection_path,res=224):
        train_dataset = RoboPointDataset(image_folder,json_detection_path,res)
        save_path=self.config["test_save_path"]  
        for i in range(len(train_dataset)-1,0,-300):
            sample=train_dataset[i]
            texts=[sample["text"]]
            images=[sample["image"]]
            tokens = self.processor(text=texts, images=images,return_tensors="pt", padding="longest")
            tokens["flag"]=[sample["flag"]]
            tokens["raw_label"]=[sample["raw_label"]]
            tokens=tokens.to(self.device)
            self.pretrained_model.eval()
            output_dict=self.pretrained_model.forward_eval(**tokens)
            print("Output_loss:",output_dict["loss"])
            print("Output_label:",sample["raw_label"])
            print("Text:",sample["text"])
            q_trans=output_dict["q_trans"].view(224,224,1).detach()
            
            x_flat = q_trans.view(-1)

            softmax_x_flat = torch.nn.functional.softmax(x_flat, dim=0)

            heatmap = softmax_x_flat.view(224, 224, 1)

            # Verify that the sum is 1
            print(torch.sum(heatmap))  
            if sample["flag"]=="detection_1":
                answer_points=ast.literal_eval(sample["raw_label"])
                assert type(answer_points[0]) is float and len(answer_points)==4
                bbox=convert_xyxy_to_cxcywh(answer_points)
                label=torch.tensor([[bbox[0],bbox[1]]])
                visualize_bboxes_and_heatmap(sample["image"], [bbox], heatmap , save_path,
                                            bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                                            bbox_width=2)  
            elif sample["flag"]=="detection_2":
                answer_points=ast.literal_eval(sample["raw_label"])
                assert type(answer_points[0]) is tuple and len(answer_points[0])==4
                labels=torch.tensor([ [answer_point[0],answer_point[1]] for answer_point in answer_points ])
                visualize_bboxes_and_heatmap(sample["image"], answer_points, heatmap, save_path,
                                            bbox_colors=['red', 'lime', 'cyan', 'yellow'],
                                            bbox_width=2)      
            else:
                assert False          



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






