# Pretrain
bash pretrain/pretrain.sh --branches 2 \
    --config_path pretrain_config.yaml \
    --json_detection_path ../outputs/pretrain_data/detection_data.json \
    --image_folder ../outputs/pretrain_data

# # RLBench Finetune
# bash finetune/RLBench/train.sh --exp_cfg_path  configs/rlbench_config.yaml \
#     --exp_note rlbench \
#     --freeze_vision_tower \
#     --log_dir ../outputs/logs/rlbench \
#     --load_pretrain \
#     --pretrain_path  ../outputs/results/pretrain/baseline_bridgevla_pretrain/20250722_150000/checkpoint-1000  