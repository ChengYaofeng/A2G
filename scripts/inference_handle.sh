#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./inference.py --task=OpenDoor \
                --task_config=cfg/open_door.yaml \
                --agent_config=cfg/agent/config.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                --model_dir='/home/cyf/task_grasp/ABCDEFG/logs/franka_open_door/model_08-13-18_epoch56.pt' \
                # --headless \

                # --test \

                
    