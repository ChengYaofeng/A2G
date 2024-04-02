#!/bin/bash

set -x
set -e

export CUDA_VISIBLE_DEVICES=0
# export PYOPENGL_PLATFORM=egl
# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python ./inference.py --task=Valve \
                --task_config=cfg/rotate_valve.yaml \
                --rl_device=cuda:0 \
                --sim_device=cuda:0 \
                --pipeline=cpu \
                --model_dir='/home/cyf/task_grasp/A-G/checkpoints/all_views/valve/force1_pi_4_model_10-03-10-43_epoch33.pt' \
                --batch_size=1 \
                --num_envs=1 \
                --headless \

    # -           --exp_parameter=force1 \

                #/home/cyf/task_grasp/A-G/logs/franka_pick_up/model_09-08-12-16_epoch36.pt
                # --test \
                # /home/cyf/task_grasp/A-G/logs/franka_rotate_valve/force1_nograv_rotate_4_pi_model_09-30-22-42_epoch16.pt

                # ALL   /home/cyf/task_grasp/A-G/logs/franka_rotate_valve/force1_pi_4_model_10-03-10-56_epoch40.pt
                

                #52
                #最好模型
                # /home/cyf/task_grasp/A-G/checkpoints/all_views/valve/force1_pi_4_model_10-03-10-43_epoch33.pt

                #差点意思
                # /home/cyf/task_grasp/A-G/logs/franka_rotate_valve/force1_all_model_10-14-17-04_epoch34.pt