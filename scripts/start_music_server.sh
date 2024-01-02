#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

WORK_DIR=$(dirname $(dirname $(readlink -f "$0")))"/real_gemini/services/"
echo "WORK_DIR: $WORK_DIR"
cd $WORK_DIR

python music_server.py \
    --host 192.168.81.13 \
    --port 6678 \
    --model_path /cognitive_comp/pankunhao/pretrained/musicgen-small

