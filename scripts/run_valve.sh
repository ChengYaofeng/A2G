#!/bin/bash

set -x
set -e
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./train.py --task=Valve \
                --task_config=cfg/rotate_valve.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                --exp_parameter=force1_all \
                --batch_size=5 \
                --num_envs=64 \
                --max_epoch=100 \
                # --headless \
                # --dataset_path='/home/cyf/task_grasp/A-G/logs/franka_rotate_valve/dataset_valve_mas0.01_force1_nograv_norotate' \

