import torch
import numpy as np
import cv2
import pickle
import argparse
import open3d as o3d
import multiprocessing as mp
from detectron2.engine import DefaultPredictor
from pcfgrasp_method.run_utils.config import load_config
from pcfgrasp_method.utils.grasp_estimator import GraspEstimatior, extract_point_clouds
from pcfgrasp_method.utils.visual_grasp import visualize_grasps

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from pcfgrasp_method.mask_rcnn.demo.predictor import VisualizationDemo
from pcfgrasp_method.mask_rcnn.demo.demo import setup_cfg
# from utils.covert_pc import depth_image_to_pc_gpu
from utils.vis_grasp import visualize_grasps_new

def pcfgrasp(bgr, depth, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar, device):
    '''
    rgb 是 bgr格式
    
    return:
        PCF-Grasp中config选择的最后的top 20的抓取姿态和成绩  {np.ndarray}  20x4x4, 20
    '''
    args_dict = {}
    args_dict['config_dir'] = '/home/cyf/task_grasp/RL_BUILDER/pcfgrasp_method/config.yaml' #pcf_config
    args_dict['use_gpu'] = False  #gpu or cpu
    args_dict['ckpt_dir'] = '/home/cyf/task_grasp/RL_BUILDER/pcfgrasp_method/checkpoints/train/03-07-08-05_best_train_94.pth'  #grasp model
    args_dict['filter'] = False   #score filter
    args_dict['pretrain_ckpt'] = '/home/cyf/task_grasp/RL_BUILDER/pcfgrasp_method/checkpoints/pretrain/03-05-14_best_pre_598.pth'
    
    clobal_cfg_load = True
    
    if clobal_cfg_load:
        global_config = load_config(args_dict['config_dir'])
        clobal_cfg_load = False

    torch.backends.cudnn.benchmark = True
    grasp_estimatior = GraspEstimatior(global_config)
    
    mp.set_start_method("spawn", force=True)
    ##### detectron mask ####
    handle_path="/home/cyf/task_grasp/RL_BUILDER/pcfgrasp_method/mask_rcnn/graspnet_mask_rcnn/output/handle/OD_cfg.pickle"
    with open(handle_path,'rb') as f:
        cfg_handle=pickle.load(f)
    #读取训练的模型生产一个预测器
    cfg_handle.MODEL.WEIGHTS="/home/cyf/task_grasp/RL_BUILDER/pcfgrasp_method/mask_rcnn/graspnet_mask_rcnn/output/handle/model_final.pth"   # handle
    cfg_handle.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5   
    predictor_handle=DefaultPredictor(cfg_handle)
    outputs_handle=predictor_handle(bgr)

    #############Visualize Mask###############11
    
    # vis=Visualizer(bgr[:,:,::-1],metadata={},scale=0.5,instance_mode=ColorMode.SEGMENTATION)
    # vis=vis.draw_instance_predictions(outputs_handle["instances"].to("cpu"))

    # WINDOW_NAME = "Handle detection"
    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # # print(visualized_output.get_image()[:, :, ::-1].shape)  #(480, 640, 3)
    # cv2.imshow(WINDOW_NAME, vis.get_image()[:, :, ::-1])
    
    # key_select = cv2.waitKey(0)
    
    ##########################################11
    mask = outputs_handle['instances'].pred_masks.cpu().numpy() * depth
    # print(mask)
    # print(mask.shape)
    # print(np.sum(mask))
    
    
    ##########################################22
    # mask_args = {}
    # mask_args['config_file'] = '/home/cyf/PCF-Grasp/pcfgrasp_method/mask_rcnn/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    # mask_args['opts'] = ['MODEL.WEIGHTS', 'detectron2://COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl']
    # mask_args['confidence_threshold'] = 0.1
    
    # cfg_load = True
    
    # if cfg_load:
    #     cfg = setup_cfg(mask_args)
    #     cfg_load = False

    # demo = VisualizationDemo(cfg)
    # predictions, visualized_output = demo.run_on_image(bgr)

    # mask = predictions['instances'].pred_masks.cpu().numpy() * depth
    # print(mask.shape)
    ##############################################22
    
    ##### vis mask ####
    # WINDOW_NAME = "Grasp detections"
    # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    # # print(visualized_output.get_image()[:, :, ::-1].shape)  #(480, 640, 3)

    # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
    # key_select = cv2.waitKey(0)
    
    pc_full = None
    # segmap = None
    # obj_pc = None
    # cam_K = np.array([[659.394531 , 0. , 320.0 ],
    #             [0. , 659.39455 , 240.0 ], 
    #             [0. ,    0. ,       1. ]])
    
    # color_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # print(camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar)
    # print(camera_view_matrix_inv)
    # tensor([[-1.6440e-01,  9.8639e-01,  7.4506e-09, -0.0000e+00],
    #     [ 5.4197e-01,  9.0328e-02,  8.3553e-01, -0.0000e+00],
    #     [ 8.2416e-01,  1.3736e-01, -5.4944e-01, -0.0000e+00],
    #     [ 2.0000e-01,  1.0000e-01,  8.0000e-01,  1.0000e+00]])
    
    obj_pc = depth_image_to_pc_gpu(torch.tensor(mask[0]), camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar).numpy()

    # pc_full = depth_image_to_pc_gpu(torch.asarray(depth), camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar).numpy()
    # if pc_full is None:
    #     pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=[0.2, 1.2])
        
    # if obj_pc is None:
    #     print('Converting depth to point cloud(s)...')
    #     obj_pc, _, obj_colors = extract_point_clouds(mask[0], cam_K, segmap=segmap, rgb=color_image, skip_border_objects=False, z_range=[0.2, 1.2])

    
    #################0819
    num_points = 100  # irrelevant point removal strategy
    radius = 0.01
    # sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_pc)
    sor_pcd, ind = pcd.remove_radius_outlier(num_points, radius)
    sor_pcd.paint_uniform_color([0, 0, 1])

    obj_pc = np.array(sor_pcd.points)
    #################
    print('Predicting grasps...')
    pred_grasps_cam, scores, idx, pred_points = grasp_estimatior.predict_scene_grasps(obj_pc, args_dict, pc_segments={}, forward_passes=1)
    # visualize_grasps_new(obj_pc, pred_grasps_cam[-1])
    # print(scores)
    # print(idx) 0-1023
    
    return pred_grasps_cam[-1], scores[-1], idx, pred_points


def depth_image_to_pc_gpu(camera_numpy, camera_view_matrix_inv, camera_proj_matrix, u, v, width, height, depth_bar=[0.2, 1.2]):
    '''
    camera_tensor
    '''
    depth_buffer = camera_numpy
    vinv = camera_view_matrix_inv #{^w_c}T^{T}
    proj = camera_proj_matrix
    
    # print(vinv, proj)
    # tensor([[-1.6440e-01,  9.8639e-01, -1.1429e-09,  0.0000e+00],
    #     [ 5.4197e-01,  9.0328e-02,  8.3553e-01,  0.0000e+00],
    #     [ 8.2416e-01,  1.3736e-01, -5.4944e-01,  0.0000e+00],
    #     [ 2.0000e-01,  1.0000e-01,  8.0000e-01,  1.0000e+00]]) 
    # tensor([[ 1.0000,  0.0000,  0.0000,  0.0000],
    #     [ 0.0000,  0.7500,  0.0000,  0.0000],
    #     [ 0.0000,  0.0000,  0.0000, -1.0000],
    #     [ 0.0000,  0.0000,  0.0010,  0.0000]])

    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2
    
    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    # print(Z)
    valid = Z < depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    # position = torch.vstack((Y, X, Z, torch.ones(len(X))))[:, valid] #这里对调xy
    position = torch.vstack((X, Y, Z, torch.ones(len(X))))[:, valid]
    
    R_coor_x = torch.tensor([[0,  -1, 0,  0],
                            [-1,  0,  0,  0],
                            [0,  0,  -1,  0],
                            [0,  0,  0,  1]], dtype=torch.float32)
    
    position = torch.matmul(R_coor_x ,position)
    
    position = position.permute(1, 0)
    # print(position.shape) #N,4
    
    points = position[:, 0:3]
    # print(torch.sum(points, dim=0))
    
    points = points[(points[:,2] < 1.2) & (points[:,2] > 0.1)] #正

    return points
