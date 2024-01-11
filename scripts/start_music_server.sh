#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

WORK_DIR=$(dirname $(dirname $(readlink -f "$0")))"/real_gemini/services/"
echo "WORK_DIR: $WORK_DIR"
cd $WORK_DIR

python music_server.py
