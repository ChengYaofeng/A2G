#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl

python ./run_tools/real_world_inference.py --pretrain_ckpt '/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/pretrain/03-05-14_best_pre_598.pth' \
                    --ckpt_dir '/home/cyf/PCF-Grasp/pcfgrasp_method/checkpoints/train/03-16-22_best_ori_16.pth' \
                    --filter False