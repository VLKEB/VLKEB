#/bin/bash
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
