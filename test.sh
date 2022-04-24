#!/bin/bash

CUDA_VISIBLE_DEVICES='0' python -u ./test.py \
        --model_name="/media/qcm/HardDisk1/code/AEC_DeepModel/checkpoints/Denoise_Baseline_epoch60_batch64_202203180928/epochno60.pth" \
        --train_data="/media/qcm/HardDisk1/jsonfile/denoise/train220316.json" \
        --val_data="/media/qcm/HardDisk1/jsonfile/denoise/eval220316.json" \
        --test_data="/media/qcm/HardDisk1/jsonfile/denoise/eval220316.json" \
        --output_dir="./test_output/"
        