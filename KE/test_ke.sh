#/bin/bash

# USAGE: bash test_ke.sh [GPU_ID] [MODEL_NAME] [CHECKPOINT_PATH]
# EXAMPLE: bash test_ke.sh 0 blip2 "/path/to/checkpoint"

# MODEL_NAME=[blip2, minigpt4, llava]

GPU=$1
MODEL=$2
CHECKPOINT=$3

export CUDA_VISIBLE_DEVICES=$GPU
time=$(date "+%Y%m%d_%H%M%S")
python test_ke.py \
    --model_checkpoint $CHECKPOINT \
    --model_name $MODEL \
    --gpus 1 \
    --num_workers 0 \
    --batch_size 1 \
    2>&1 | tee models/$MODEL/$time\_test_log_$MODEL.txt
