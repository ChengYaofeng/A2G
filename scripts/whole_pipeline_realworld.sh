#!/bin/bash

set -x
set -e

python ./run_tools/whole_pipeline_realworld.py \
        --a_grasp_task pickup \
        --a_grasp_ckp_path '/home/franka/cg_ws/src/contact_graspnet_ros/src/a2g/checkpoints/pick_up/mid/model_10-10-02_epoch86.pt' \
        --exp 'wo_pose' #wo_pose # tag #to  #wo_points #wo_gpf

        #valve 
        # /home/franka/cg_ws/src/contact_graspnet_ros/src/a2g/checkpoints/valve/force1_pi_4_model_10-03-10-43_epoch33.pt

        #handle
        # /home/franka/cg_ws/src/contact_graspnet_ros/src/a2g/checkpoints/open_door/model_10-06-20_epoch97.pt


        # /home/franka/cg_ws/src/contact_graspnet_ros/src/a2g/checkpoints/open_door/pick_mas0.15_allview_graspedge_model_10-19-10-19_epoch14.pt

        # pickup

        # single pick
        # /home/franka/cg_ws/src/contact_graspnet_ros/src/a2g/checkpoints/pick_up/mid/model_10-10-02_epoch86.pt

        # depth layer
        # /home/franka/cg_ws/src/contact_graspnet_ros/src/a2g/checkpoints/pick_up/1014 model single/single_pick_10-14-15-46_epoch97.pt

        # double pick
        # /home/franka/cg_ws/src/contact_graspnet_ros/src/a2g/checkpoints/pick_up/pick_mas0.15_model_09-25-10-34_epoch50.pt       #zhege