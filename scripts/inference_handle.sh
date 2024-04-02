#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./inference.py --task=OpenDoor \
                --task_config=cfg/open_door.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                --model_dir='/home/cyf/task_grasp/A-G/logs/franka_open_door/test_loss_pn_model_11-09-20-11_epoch46.pt' \
                --batch_size=1 \
                --num_envs=1 \
                --headless \

                
    