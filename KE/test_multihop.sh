#/bin/bash

# USAGE: bash test_multihop.sh [GPU_ID] [MODEL_NAME] [HOP_NUM]
# EXAMPLE: bash test_multihop.sh 0 blip2 2

# MODEL_NAME=[blip2, minigpt4, llava]
# HOP_NUM=[1, 2, 3, 4]

Blip2_ckpt="models/blip2/version_82/checkpoints/model-epoch=04-valid_acc=0.9451_clean.ckpt"
Mini_ckpt="models/minigpt4/version_15/checkpoints/model-epoch=03-valid_acc=0.9848_clean.ckpt"
Llava_ckpt="models/llava/version_20/checkpoints/model-epoch=04-valid_acc=0.9907_clean.ckpt"

GPU=$1
MODEL=$2
HOP=$3

if [ $MODEL == "blip2" ]; then
    CHECKPOINT=$Blip2_ckpt
elif [ $MODEL == "minigpt4" ]; then
    CHECKPOINT=$Mini_ckpt
elif [ $MODEL == "llava" ]; then
    CHECKPOINT=$Llava_ckpt
else
    echo "Invalid model name, choose from [blip2, minigpt4, llava]"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU
time=$(date "+%Y%m%d_%H%M%S")
python test_multihop.py \
    --model_name $MODEL \
    --hop $HOP \
    --model_checkpoint $CHECKPOINT \
    --gpus 1 \
    --num_workers 0 \
    --batch_size 1 \
    2>&1 | tee models/$MODEL/$time\_test_log_$MODEL.txt
