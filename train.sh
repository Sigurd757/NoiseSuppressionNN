#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python -u ./train.py \
        --batch_size=8 \
        --epochs=60 \
        --lr=1e-3 \
        --train_data="/media/qcm/HardDisk1/jsonfile/denoise/train220316.json" \
        --val_data="/media/qcm/HardDisk1/jsonfile/denoise/eval220316.json" \
        --checkpoints_dir="./checkpoints" \
        --event_dir="./eventfile/Denoise_baseline" \
        --min_lr=1e-6 
        
