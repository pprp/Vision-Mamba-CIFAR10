# Vision-mamba-CIFAR

## Introduction

`Vision-mamba-CIFAR` is a toy project focused on image classification tasks using the CIFAR-10 dataset. Built with PyTorch and leveraging the state-of-the-art models from the `timm` library, this project aims to provide a high-level interface for training, evaluating, and deploying models trained on CIFAR-10. Whether you're a researcher, student, or hobbyist, `Vision-mamba-CIFAR` offers an accessible yet powerful way to jumpstart your projects in computer vision. Most of code is from `Vim`: https://github.com/hustvl/Vim 

## Features

- Pre-trained model support from `timm`.
- For the study of Mamba

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- torchvision
- timm

### Setup

Clone the repository to your local machine:

```bash
git clone git@github.com:pprp/Vision-Mamba-CIFAR10.git
cd Vision-mamba-CIFAR
```

### Installation 

```
pip install -r requirements
```

## Training 

```
CUDA_VISIBLE_DEVICES=6 python main.py --data-set CIFAR \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_rope_also_residual --data-path ./data \
    --epochs 200 --lr 5e-4 \
    --batch-size 640 --gpu 6
```

```
bash run.sh
```