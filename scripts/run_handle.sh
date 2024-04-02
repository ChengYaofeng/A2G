#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./train.py --task=OpenDoor \
                --task_config=cfg/open_door.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                --exp_parameter=test_loss_3_256 \
                --batch_size=5 \
                --num_envs=128 \
                --max_epoch=100 \
                --headless \
                --dataset_path='/home/cyf/task_grasp/A-G/logs/franka_pick_up/dataset_1014_mas0.2_viewleft_useful' \

