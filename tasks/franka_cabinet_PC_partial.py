from isaacgym.torch_utils import *
import numpy as np
from tasks.open_cabinet import OpenCabinet
import matplotlib.pyplot as plt
from pointnet2_ops import pointnet2_utils
import os
from run_utils.register import TASKS

@TASKS.register('open_cabinet_pc_part')
class OneFrankaCabinetPCPartial(OpenCabinet):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, 
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, log_dir=None):
        
        self.cabinetPCOriginalNum = cfg["env"]["cabinetPointOriginalNum"]
        self.cabinetPCDownsampleNum = cfg["env"]["cabinetPointDownsampleNum"]
        self.handPCDownsampleNum = cfg["env"]["handPointDownsampleNum"]
        self.pointCloudDownsampleNum = self.cabinetPCDownsampleNum + self.handPCDownsampleNum    #采集了手和柜子的点云
        if not hasattr(self, "cabinet_mask_dim"):
            self.cabinet_mask_dim = 4
        
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless, log_dir=log_dir)   
        #注意在这里出现了obs的第一次定义 & cabinet_pc的定义

        self.num_obs += 5  #加5的原因是，其中3是点云，1是mask，1是map_score
        self.task_meta["mask_dim"] = 2
        self.task_meta["obs_dim"] = self.num_obs
        self.num_feature = cfg["env"]["pointFeatureDim"]
        self.obs_buf = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, self.num_obs), device=self.device, dtype=torch.float)
        # print("obs_buf shape-------------: ", self.obs_buf.shape)   #torch.Size([1, 2112, 60]) 55+5
        
        if cfg["env"]["visualizePointcloud"] == True :
            import open3d as o3d
            from utils.o3dviewer import PointcloudVisualizer
            self.pointCloudVisualizer = PointcloudVisualizer()
            self.pointCloudVisualizerInitialized = False
            self.o3d_pc = o3d.geometry.PointCloud()
        else :
            self.pointCloudVisualizer = None

        self.cabinet_pc = self.cabinet_pc.repeat_interleave(self.env_per_cabinet, dim=0)   #torch.Size([1, 8192, 4])
        # print(self.cabinet_pc.shape)
        
        self.sampled_cabinet_pc = torch.zeros((self.num_envs, self.cabinetPCDownsampleNum, self.cabinet_mask_dim), device=self.device)
        #这个是采样后的点，压缩到了2048
        self.selected_cabinet_pc = self._detailed_view(self.cabinet_pc)[:, 0, ...]    #这么做的目的应该是只取第一个的柜子作为对象
        # print(self._detailed_view(self.cabinet_pc).shape)  #torch.Size([1, 1, 8192, 4])
        # print(self.selected_cabinet_pc.shape)  #torch.Size([1, 8192, 4])
        self._refresh_map()

    def _refresh_map(self):

        self.sampled_cabinet_pc = self.sample_points(
            self.selected_cabinet_pc,
            self.cabinetPCDownsampleNum,
            sample_method="furthest"
        ).repeat_interleave(self.env_per_cabinet, dim=0)
        
        # print(self.sampled_cabinet_pc.shape)

    def _get_transformed_pc(self, pc=None, mask=None):

        if pc is None:
            pc = self.selected_cabinet_pc[:, :, :3]
        if mask is None:
            mask = self.selected_cabinet_pc[:, :, 3]

        # select first env of each type of cabinet
        used_initial_root_state = self._detailed_view(self.initial_root_states)[:, 0, ...]
        used_initial_rigd_state = self._detailed_view(self.initial_rigid_body_states)[:, 0, ...]

        transformed_pc = self._transform_pc(
            pc,
            mask.view(self.cabinet_num, -1, 1),
            used_initial_root_state[:, 1, :7],
            used_initial_rigd_state[:, self.cabinet_rigid_body_index, :7]
        )

        return transformed_pc
    
    def get_map(self, raw_point_clouds, raw_mask, raw_buffer_list):
        '''
        这里对应的就是论文中的MPR
        '''

        map_dis_bar = self.cfg['env']['map_dis_bar']
        top_tensor = torch.tensor([x.top for x in self.contact_buffer_list], device=self.device)
        buffer_size = (raw_buffer_list[-1]).buffer.shape[0]

        buffer = torch.zeros((self.cabinet_num, buffer_size, 3)).to(self.device)
        for i in range(self.cabinet_num):
            buffer[i] = raw_buffer_list[i].buffer[:, 0:3]

        dist_mat = torch.cdist(raw_point_clouds, buffer, p=2)
        if_eff = dist_mat<map_dis_bar
        
        for i in range(self.cabinet_num):
            if_eff[i, :, top_tensor[i]:] = False

        tot = if_eff.sum(dim=2) * raw_mask
        # tot = torch.log(tot+1)
        tot_scale = tot/(tot.max()+1e-8)  # env*pc
        # tot_scale = tot_scale * raw_mask

        return tot_scale
    
    
    def save(self, path, iteration):

        self.raw_map = self.get_map(self.selected_cabinet_pc[:, :, :3], self.selected_cabinet_pc[:, :, 3], self.contact_buffer_list)
        transformed_pc = self._get_transformed_pc()
        saving_raw_map = torch.cat((transformed_pc[:, :, :3], self.raw_map[:, :].view(self.cabinet_num, -1, 1)), dim=-1)
        torch.save(saving_raw_map, os.path.join(path, "rawmap_{}.pt".format(iteration)))
        
    
    def _transform_pc(self, pc, moving_mask, fixed_seven, moving_seven):
        '''
        目的是把点云转换到世界坐标系下
        Input:
            pc: (env, pc, 3)
            moving_mask: (env, pc, 1)
            fixed_seven: (env, 7)
            moving_seven: (env, 7)
        '''

        fixed_p = fixed_seven[:, :3]
        fixed_r = fixed_seven[:, 3:7]
        moving_p = moving_seven[:, :3]
        moving_r = moving_seven[:, 3:7]

        shape = pc.shape
        
        fixed_point_clouds = quat_apply(fixed_r.view(-1, 1, 4).repeat_interleave(shape[1], dim=1), pc[:, :, :3]) + fixed_p.view(-1, 1, 3)
        moving_point_clouds = quat_apply(moving_r.view(-1, 1, 4).repeat_interleave(shape[1], dim=1), pc[:, :, :3]) + moving_p.view(-1, 1, 3)
        merged_point_clouds = torch.where(moving_mask>0.5, moving_point_clouds, fixed_point_clouds)

        return merged_point_clouds

    def compute_point_cloud_state(self, pc=None):
        '''
        这里的作用？
        '''

        lfinger_p = self.rigid_body_tensor[:, self.hand_lfinger_rigid_body_index, :3]
        lfinger_r = self.rigid_body_tensor[:, self.hand_lfinger_rigid_body_index, 3:7]
        rfinger_p = self.rigid_body_tensor[:, self.hand_rfinger_rigid_body_index, :3]
        rfinger_r = self.rigid_body_tensor[:, self.hand_rfinger_rigid_body_index, 3:7]

        if pc == None :
            selected_pc = self.sample_points(self.cabinet_pc, self.cabinetPCDownsampleNum, sample_method='furthest')
        else :
            selected_pc = pc
        
        # print(selected_pc.shape)  #torch.Size([1, 2048, 5])
        
        selected_lfinger_pc = self.sample_points(self.franka_left_finger_pc, self.handPCDownsampleNum//2, sample_method='furthest')
        selected_rfinger_pc = self.sample_points(self.franka_right_finger_pc, self.handPCDownsampleNum//2, sample_method='furthest')
        # print(selected_lfinger_pc.shape)  #torch.Size([1, 32, 3])

        lfinger_shape = selected_lfinger_pc.shape
        rfinger_shape = selected_rfinger_pc.shape
        moving_mask = selected_pc[:, :, 3].view(self.num_envs, self.cabinetPCDownsampleNum, 1) #这里把mask选出来
        pc_masks = selected_pc[:, :, 3:].view(self.num_envs, self.cabinetPCDownsampleNum, -1) #这里的作用是什么 torch.Size([1, 2048, 2]) mask和成绩为什么都要
        mask_num = pc_masks.shape[-1]
        # print(pc_masks.shape)  #torch.Size([1, 2048, 2])
        # print(mask_num)  #2
        finger_mask = torch.tensor([0]*mask_num+[1], device=self.device)  #这里mask是 0 0 1 的作用
        # print(finger_mask)  #tensor([0, 0, 1], device='cuda:0')

        #这里的目的是为了实现夹爪点云在环境中的对应
        lfinger_point_clouds = quat_apply(lfinger_r.view(-1, 1, 4).repeat_interleave(lfinger_shape[1], dim=1), selected_lfinger_pc) + lfinger_p.view(-1, 1, 3)
        rfinger_point_clouds = quat_apply(rfinger_r.view(-1, 1, 4).repeat_interleave(rfinger_shape[1], dim=1), selected_rfinger_pc) + rfinger_p.view(-1, 1, 3)
        merged_point_clouds = self._transform_pc(selected_pc[:, :, :3], moving_mask, self.root_tensor[:, 1, :7], self.rigid_body_tensor[:, self.cabinet_rigid_body_index, :7])
        # print(merged_point_clouds)  #torch.Size([1, 2048, 3]   [-0.1717,  0.4865,  1.2458]  移动后的cabinet的点云
        
        # print(lfinger_point_clouds) #torch.Size([1, 32, 3])  [ 1.1687e-02,  2.6738e-02,  1.4596e+00]
        merged_point_clouds = append_mask(torch.cat((merged_point_clouds, pc_masks), dim=-1), torch.tensor([0], device=self.device))   #最后的纬度就是0
        lfinger_point_clouds = append_mask(lfinger_point_clouds, finger_mask)
        rfinger_point_clouds = append_mask(rfinger_point_clouds, finger_mask)
        # print(lfinger_point_clouds) #torch.Size([1, 32, 6])  后面的三纬度就是 0 0 1 [ 1.1687e-02,  2.6738e-02,  1.4596e+00,  0.0000e+00,  0.0000e+00, 1.0000e+00]
        # print(merged_point_clouds)  #torch.Size([1, 2048, 6]  [-0.1717,  0.4865,  1.2458,  1.0000,  0.5064,  0.0000]  这里推测最后一纬度为0的目的是mask，为了把夹爪的点云区分开
        
        point_clouds = torch.cat((merged_point_clouds, lfinger_point_clouds, rfinger_point_clouds), dim=1)

        return point_clouds


    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    # @TimeCounter
    def sample_points(self, points, sample_num=1000, sample_method='random', sample_prob=None):
        eff_points = points
        if sample_method == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_method == 'furthest':
            idx = pointnet2_utils.furthest_point_sample(points[:, :, :3].contiguous().cuda(), sample_num).long().to(self.device)
            idx = idx.view(*idx.shape, 1).repeat_interleave(points.shape[-1], dim=2)
            sampled_points = torch.gather(points, dim=1, index=idx)
        elif sample_method == 'edge' :
            if sample_prob == None :
                sample_prob = torch.ones((eff_points.shape[0], eff_points.shape[1]), device=self.device)
            # idx = torch.topk(sample_prob, sample_num, dim=-1).indices
            idx = torch.multinomial(sample_prob, sample_num)
            idx = idx.view(*idx.shape, 1).repeat_interleave(points.shape[-1], dim=2)
            sampled_points = torch.gather(points, dim=1, index=idx)
        return sampled_points
    
    def _refresh_pointcloud_visualizer(self, point_clouds, data):

        if isinstance(point_clouds, list) :
            points = np.concatenate([a.cpu().numpy() for a in point_clouds], axis=0)
        else :
            points = point_clouds.cpu().numpy()
        
        if isinstance(data, list):
            colors = np.concatenate([a.cpu().numpy() for a in data], axis=0)
        else :
            colors = data.cpu().numpy()

        import open3d as o3d
        colors = plt.get_cmap()(colors)[:, :3]
        self.o3d_pc.points = o3d.utility.Vector3dVector(points)
        self.o3d_pc.colors = o3d.utility.Vector3dVector(colors)

        if self.pointCloudVisualizerInitialized == False :
            self.pointCloudVisualizer.add_geometry(self.o3d_pc)
            self.pointCloudVisualizerInitialized = True
        else :
            self.pointCloudVisualizer.update(self.o3d_pc)
    
    def _refresh_observation(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg["env"]["driveMode"] == "ik" :
            self.gym.refresh_jacobian_tensors(self.sim)

        # print('self.sampled_cabinet_pc', self.sampled_cabinet_pc.shape)    #torch.Size([2, 2048, 4])
        point_clouds = self.compute_point_cloud_state(pc=self.sampled_cabinet_pc)
        # print('point_clouds:', point_clouds.shape)   #torch.Size([2, 2112, 5])

        point_clouds[:, :, :3] -= self.franka_root_tensor[:, :3].view(-1, 1, 3)
        self.obs_buf[:, :, :5].copy_(point_clouds)   #这里用点云填充了前5列
        self.obs_buf[:, :, 5:] = self._get_base_observation().view(self.num_envs, 1, -1).repeat_interleave(self.pointCloudDownsampleNum, dim=1)

        if self.pointCloudVisualizer != None :
            self._refresh_pointcloud_visualizer(
                [point_clouds[0, :, :3], self.contact_buffer_list[0].all()[:, :3]],
                [point_clouds[0, :, 3], torch.ones((self.contact_buffer_list[0].top,)) * 0.5]
            )


# HELPER FUNCTIONS

def append_mask(x, mask):

    return torch.cat((x, mask.view(1, 1, -1).repeat_interleave(x.shape[0], dim=0).repeat_interleave(x.shape[1], dim=1)), dim=2)

'''
This function is used to compute realtime pointcloud.
However, Isaacgym didn't support realtime pointcloud capturing very well,
from our observation, Isaacgym will not process multiple captures in parallel,
so we end up using simulated pointcloud, not realtime pointcloud.
'''
@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
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