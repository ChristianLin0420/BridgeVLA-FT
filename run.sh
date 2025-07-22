# Pretrain
bash pretrain/pretrain.sh --branches 2 \
    --config_path pretrain_config.yaml \
    --json_detection_path ../outputs/pretrain_data/detection_data.json \
    --image_folder ../outputs/pretrain_data

# # RLBench Finetune
# cd ../finetune/RLBench
# bash train.sh --exp_cfg_path  configs/rlbench_config.yaml \
#               --exp_note rlbench \
#               --freeze_vision_tower \
#               --log_dir /localhome/local-chrislin/vla/BridgeVLA-FT/outputs/logs/rlbench \
#               --load_pretrain \
#               --pretrain_path  /localhome/local-chrislin/vla/BridgeVLA-FT/outputs/results/pretrain/checkpoint-1000