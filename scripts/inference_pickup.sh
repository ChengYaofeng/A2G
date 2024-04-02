#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./inference.py --task=PickUp \
                --task_config=cfg/pickup_obj.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                --model_dir='/home/cyf/task_grasp/A-G/logs/00_useful_model_history/pickup/1014 model single/pick_mas0.15_allview_graspedge_model_10-14-15-46_epoch92.pt' \
                --batch_size=1 \
                --num_envs=1 \
                --headless \

                
    