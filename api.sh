#!/bin/bash
source activate llama
cd /root/LLaMA-Factory

CUDA_VISIBLE_DEVICES=0 API_PORT=6006 python src/api.py \
    --model_name_or_path /root/autodl-tmp/internlm2_5-7b-chat \
    --template intern2 \
