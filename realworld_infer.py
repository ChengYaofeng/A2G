import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, 'tasks'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'agents'))
sys.path.append(os.path.join(BASE_DIR, 'assets'))
sys.path.append(os.path.join(BASE_DIR, 'collision_predictor'))
sys.path.append(os.path.join(BASE_DIR, 'pcfgrasp_method'))

sys.path.append(os.path.join(BASE_DIR, 'agents', 'ppo'))

from run_utils.parser import get_args, gen_sim_params, set_seed
from run_utils.config import load_config
import argparse
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import multiprocessing as mp
import torch
import yaml

from pcfgrasp_method.utils.grasp_estimator import extract_point_clouds
from agents.task_score_model import TaskScoreModel
from utils.visual_point import vis_score_pc

from pcfgrasp_method.mask_rcnn.demo.demo import get_parser, setup_cfg
from pcfgrasp_method.mask_rcnn.demo.predictor import VisualizationDemo
from pcfgrasp_method.mask_rcnn.detectron2.engine import DefaultPredictor

def main(args, cfg_train, z_range=[0.2,1.2]):

    align = rs.align(rs.stream.color) #深度图和rgb图对齐

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    file_num = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            # depth_frame = rs.hole_filling_filter().process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image1 = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            cv2.namedWindow('color image', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('color image', cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1)

            # print("FPS = {0}".format(int(1/(time_end-time_start))))

            # press ' ' to save current RGBD images and pointcloud.
            segmap = None
            pc_full = None
            pc_colors = None
            obj_pc = None

            if key & 0xFF == ord(' '):
                rgb = color_image1
                depth = depth_image / 1000.0
                cam_K = np.array([[intr.fx, 0, intr.ppx],
                                [0, intr.fy, intr.ppy], 
                                [0,    0,       1]])
                
                mp.set_start_method("spawn", force=True)
                # argparse = get_parser().parse_args()
                args.config_file = '/home/franka/cg_ws/src/contact_graspnet_ros/src/completion_method/mask_rcnn/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
                args.opts = ['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl']
                cfg = setup_cfg(args)

                demo = VisualizationDemo(cfg)
                predictions, visualized_output = demo.run_on_image(rgb)
                WINDOW_NAME = "Grasp detections"
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                # print(visualized_output.get_image()[:, :, ::-1].shape)  #(480, 640, 3)

                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if pc_full is None:
                    print('Converting depth to point cloud(s)...')
                    pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=z_range)

                mask = predictions['instances'].pred_masks.cpu().numpy() * depth

                # select object to generate grasp
                key_select = cv2.waitKey(0)
                if key_select & 0xFF == ord('0'):
                    idc = 0
                elif key_select & 0xFF == ord('1'):
                    idc = 1
                elif key_select & 0xFF == ord('2'):
                    idc = 2
                elif key_select & 0xFF == ord('3'):
                    idc = 3
                elif key_select & 0xFF == ord('4'):
                    idc = 4
                elif key_select & 0xFF == ord('5'):
                    idc = 5
                elif key_select & 0xFF == ord('6'):
                    idc = 6
                elif key_select & 0xFF == ord('7'):
                    idc = 7
                elif key_select & 0xFF == ord('8'):
                    idc = 8
                elif key_select & 0xFF == ord('9'):
                    idc = 9
                else:
                    print('plece input number 0-9 object to select')
                    raise Exception('Wrong Input', key_select-48)
                
                obj_num = predictions['instances'].pred_classes.tolist()
                if idc > len(obj_num):
                    raise Exception('Input out of bounds', key_select-48)
                
                if obj_pc is None:
                    print('Converting depth to point cloud(s)...')
                    obj_pc, _, obj_colors = extract_point_clouds(mask[idc], cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=z_range)
                
                # if pc_full is None:
                #     print('Converting depth to point cloud(s)...')
                #     pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=z_range)
                # print(pc_colors.shape)

                model_cfg = cfg_train['policy']

                model = TaskScoreModel(model_cfg)
               
                model.load_state_dict(torch.load(args.path, map_location='cuda'))
        
                model.eval()
                with torch.no_grad():
                    
                    ###
                    #obj_pc 
                    task_scores = model(obj_pc)    #B, N, 1
                    print('task_scores', torch.max(task_scores))
                    print('task_scores', task_scores)
                    
                    vis_score_pc(obj_pc, task_scores)
    finally:
        pipeline.stop()
                

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_train', type=str, default="cfg/agent/config.yaml")
    parser.add_argument('--path', type=str, default="/home/cyf/task_grasp/ABCDEFG/logs/franka_open_door/model_08-13-18_epoch56.pt")
    
    args = parser.parse_args()
    #load cfg
    
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:   #训练模型的，通用
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    main(args, cfg_train)