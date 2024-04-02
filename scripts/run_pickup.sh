#!/bin/bash

set -x
set -e
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./train.py --task=PickUp \
                --task_config=cfg/pickup_obj.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                --exp_parameter=pick_mas0.15_allview_graspedge \
                --batch_size=3 \
                --num_envs=256 \
                --max_epoch=100 \
                --headless \
                --dataset_path='/home/cyf/task_grasp/A-G/logs/franka_pick_up/dataset_1008_mas0.4_tight_view' \

                # --test \

                
    