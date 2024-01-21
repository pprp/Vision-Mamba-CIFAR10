#!/bin/bash 

CUDA_VISIBLE_DEVICES=6 python main.py --data-set CIFAR \
    --model vim_tiny_patch16_224 --data-path ./data \
    --epochs 200 --lr 0.1 \
    --batch-size 128 --gpu 6