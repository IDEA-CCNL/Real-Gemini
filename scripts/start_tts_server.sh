#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

WORK_DIR=$(dirname $(dirname $(readlink -f "$0")))"/real_gemini/services/"
echo "WORK_DIR: $WORK_DIR"
cd $WORK_DIR

python tts_server.py
