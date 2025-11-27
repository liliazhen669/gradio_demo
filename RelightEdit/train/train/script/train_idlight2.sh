#!/bin/bash

# usage: ./train.sh [CONFIG_FILE] [PORT]
# example: ./train.sh normal_lora.yaml 41353

CONFIG_FILE=${1:-"moe_idlight2.yaml"}
PORT=${2:-41353}

export XFL_CONFIG=./train/config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.
#CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 accelerate launch --main_process_port ${PORT} -m src.train.train_idlight
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.train.train_idlight