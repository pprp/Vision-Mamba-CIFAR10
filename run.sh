#!/bin/bash 

CUDA_VISIBLE_DEVICES=6 python main.py --data-set CIFAR \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual \
    --data-path ./data \
    --epochs 200 --lr 5e-4 \
    --batch-size 64 --gpu 6