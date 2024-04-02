import numpy as np
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
import torch
from scipy.spatial.transform import Rotation


def quaternion_to_rotation_matrix(q):
    # 将四元数转换为旋转矩阵
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = torch.stack([
        2*q1*q3 + 2*q0*q2, 2*q1*q2 - 2*q0*q3, 2*q2**2 +2*q3**2-1
    ], dim=1)
    return R

def quat_axis(q, axis=0):
    """ 提取特定的轴 0x 1z 2y """
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def relative_pose(src, dst) :
    # 这个函数的作用是
    shape = dst.shape
    p = dst.view(-1, shape[-1])[:, :3] - src.view(-1, src.shape[-1])[:, :3]
    ip = dst.view(-1, shape[-1])[:, 3:]
    ret = torch.cat((p, ip), dim=1)
    return ret.view(*shape)

def rot2quat(rotation):
    rotation = Rotation.from_matrix(rotation.cpu().numpy())
    quat = rotation.as_quat()
    return quat

def rotate_vector_back(v_rotated, q):
    R = quaternion_to_rotation_matrix(q)
    print(R.shape)
    v_inverse_rotated = torch.matmul(R.transpose(1, 2), v_rotated.unsqueeze(-1)).squeeze(-1)
    return v_inverse_rotated


def quat2rot(quat):
    """
    将四元数转换为旋转矩阵
    
    参数：
    - quat: 四元数 [w, x, y, z]
    
    返回值：
    - rot: 旋转矩阵 3x3
    """
    x, y, z, w = quat
    rot = np.zeros((3, 3))
    
    rot[0, 0] = 1 - 2 * (y**2 + z**2)
    rot[0, 1] = 2 * (x*y - w*z)
    rot[0, 2] = 2 * (x*z + w*y)
    
    rot[1, 0] = 2 * (x*y + w*z)
    rot[1, 1] = 1 - 2 * (x**2 + z**2)
    rot[1, 2] = 2 * (y*z - w*x)
    
    rot[2, 0] = 2 * (x*z - w*y)
    rot[2, 1] = 2 * (y*z + w*x)
    rot[2, 2] = 1 - 2 * (x**2 + y**2)
    
    return rot

def rotate_y(angle):
    """
    绕Y轴旋转给定角度的旋转矩阵
    
    参数：
    - angle: 旋转角度（弧度）
    
    返回值：
    - rot: 旋转矩阵 3x3
    """
    c = np.cos(angle)
    s = np.sin(angle)
    
    rot = np.array([[c, 0, -s],
                    [0, 1, 0],
                    [s, 0, c]])
    
    return rot