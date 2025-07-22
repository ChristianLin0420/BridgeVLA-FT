cd  pretrain
port=29500
GPUS_PER_NODE=1
NNODES=1
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$port \
    pretrain.py "$@"