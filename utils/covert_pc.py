import torch
import numpy as np
import copy

def depth_image_to_pc_gpu(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    depth_buffer = camera_tensor.to(device)
    vinv = camera_view_matrix_inv
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points


def center_pc_convert_cam(batch_data):
    '''
        将点云转移到质心附近，为了更好的收敛
        Input: 
            cam_poses: [B, 4, 4]
            batch_data: [B, N, 3]
        Return:
            batch_data_new: [B, N, 3]
    '''
    # print(type(cam_poses))
    # print(type(batch_data))
    
    # cam_poses[:3,1] = -cam_poses[:3,1]
    # cam_poses[:3,2] = -cam_poses[:3,2]
    # cam_poses_new = copy.deepcopy(cam_poses)
    # print(batch_data.shape)
    pc_mean = torch.mean(batch_data, dim=1, keepdims=True)
    # print(' pc_mean.shape',  pc_mean.shape)
    batch_data_new = batch_data[:, :, :3] - pc_mean[:, :, :3] #B N 3
    # print(' batch_data_new.shape',  batch_data_new.shape)
    # cam_poses[:3, 3] -= pc_mean[0, :, :3]
    # print(' batch_data_new.shape',  batch_data_new.shape)
    return batch_data_new#, cam_poses