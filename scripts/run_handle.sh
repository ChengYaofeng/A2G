#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./train.py --task=OpenDoor \
                --task_config=cfg/open_door.yaml \
                --agent_config=cfg/agent/config.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                # --headless \

                # --test \
                # --model_dir='/home/cyf/task_grasp/RL_BUILDER/logs/franka_cabinet/ppo/ppo_seed1/model_16800.pt'  #70.9

                
    