from run_utils.register import TASKS
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import *
from base import BaseTask
from PIL import Image as im
from tqdm import tqdm
import numpy as np
import torch
import os
from random import shuffle
import time
# from utils.torch_utils import to_torch
# import open3d as o3d
from utils.visual_point import vis_pc
from utils.pcf_inference import pcfgrasp
import cv2
from scipy.spatial.transform import Rotation

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
    # print(rotation)
    rotation = Rotation.from_matrix(rotation.cpu().numpy())
    quat = rotation.as_quat()
    return quat


@TASKS.register('OpenDoor')
class OpenDoor(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, 
                 headless, log_dir=None
    ):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.log_dir = log_dir
        self.up_axis = 'z'
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        self.device_type = device_type
        self.device_id = device_id
    
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)
        else:
            self.device = "cpu"
        
        # 环境数量
        self.env_num_train = cfg["env"]["numTrain"]
        self.env_num_val = cfg["env"]["numVal"]
        self.env_num = self.env_num_train + self.env_num_val
        
        self.env_ptr_list = []
        
        self.asset_root = cfg["env"]["asset"]["assetRoot"]
        self.exp_name = cfg['env']["env_name"]
        
        #TODO 之后可以把这里的参数放到config文件中
        self.camera_properties = gymapi.CameraProperties()
        self.camera_properties.width = 480
        self.camera_properties.height = 640
        self.camera_properties.enable_tensors = True
        
        # obj加载
        self.franka_loaded = False
        self.door_loaded = False
        self.camera_loaded = False
        # 采样点数量 由于采样点是通过抓取来生成的，所以抓取网络训练的结果就是这里采样的结果
        
        super().__init__(cfg=self.cfg, enable_camera_sensors=cfg['env']['enableCameraSensors'])
        #这里在base初始化的时候，会调用create_sim，由于父类里的create_sim还没有执行到，所以在这里会执行子类的create_sim
         
        # acquire tensors
        self.root_tensor = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        # print("root_tensor.shape: ", self.root_tensor.shape) #4,13
        self.dof_state_tensor = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim))
        
        # print("dof_state_tensor.shape: ", self.dof_state_tensor.shape) #20,2
        self.rigid_body_tensor = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim))
        # print("rigid_body_tensor.shape: ", self.rigid_body_tensor.shape) #26,13
        
        #if self.cfg["env"]["driveMode"] == "ik":    # inverse kinetic needs jacobian tensor, other drive mode don't need
        self.jacobian_tensor = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        # print(self.jacobian_tensor)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.jacobian_tensor)
        # print(self.jacobian_tensor)
            # print("jacobian_tensor.shape: ", self.jacobian_tensor.shape) #num_envs, 6, 9
            
        franka_link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
        franka_hand_index = franka_link_dict["panda_hand"]
        self.j_eef = self.jacobian_tensor[:, franka_hand_index - 1, :, :7].to(self.device)
            
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            
        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env, 6)  #这里必须是wrap_tensor才能在后续进行更新
        # print("vec_sensor_tensor.shape: ", self.vec_sensor_tensor.shape) #num_envs, 2, 6
        self.sensor_forces = self.vec_sensor_tensor[..., 0:3]
        # print("sensor_forces.shape: ", self.sensor_forces.shape) #num_envs, 2, 3
        self.sensor_torques = self.vec_sensor_tensor[..., 3:6]
        
        #这里是初始化的第一次刷新
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.damping = 0.05
        
        #初始状态，用来后面做reward
        self.initial_dof_states = self.dof_state_tensor.clone()
        
        # print("initial_dof_states.shape: ", self.initial_dof_states.shape) #20,2
        self.initial_root_states = self.root_tensor.clone()
        self.initial_rigid_body_states = self.rigid_body_tensor.clone().to(self.device)  #在点云的计算中用到，cp_map中用到 transform_pc
        # print(self.rigid_body_tensor.shape)
        
        # params for success rate 这里在后面计算reward中用到，这里提前写出来，以用作第一次的reset
        # self.success = torch.zeros((self.env_num,), device=self.device)
        # self.success_rate = torch.zeros((self.env_num,), device=self.device)
        # self.success_buf = torch.zeros((self.env_num,), device=self.device).long()
        
        # self.sample_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
        
        
        self.hand_body_idxs = []
        self.lfg_body_idxs = []
        self.rfg_body_idxs = []
        self.door_dof_idxs = []
        self.handle_body_idxs = []
        for i in range(self.env_num):
            hand_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.env_ptr_list[i],
                self.franka_actor_list[i],
                "panda_hand",
                gymapi.DOMAIN_SIM
            )
            hand_lfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.env_ptr_list[i],
                self.franka_actor_list[i],
                "panda_leftfinger",
                gymapi.DOMAIN_SIM
            )
            hand_rfinger_rigid_body_index = self.gym.find_actor_rigid_body_index(
                self.env_ptr_list[i],
                self.franka_actor_list[i],
                "panda_rightfinger",
                gymapi.DOMAIN_SIM
            )
            door_dof_index = self.gym.find_actor_dof_index(
                self.env_ptr_list[i],
                self.door_actor_list[i],
                self.door_dof_name,
                gymapi.DOMAIN_SIM
            )  #门把手关节
            
            handle_body_index = self.gym.find_actor_rigid_body_index(
                self.env_ptr_list[i],
                self.franka_actor_list[i],
                "link_0",
                gymapi.DOMAIN_SIM
            )
            self.hand_body_idxs.append(hand_rigid_body_index)
            self.lfg_body_idxs.append(hand_lfinger_rigid_body_index)
            self.rfg_body_idxs.append(hand_rfinger_rigid_body_index)
            self.door_dof_idxs.append(door_dof_index)
            self.handle_body_idxs.append(handle_body_index)
        
        self.env_dof_num = self.gym.get_env_dof_count(self.env_ptr_list[0])
        
        self.rb_states = self.rigid_body_tensor.to(self.device)
        self.dof_states = self.dof_state_tensor.to(self.device)
        self.init_pos = self.initial_rigid_body_states[self.hand_body_idxs, :3]
        self.init_rot = self.initial_rigid_body_states[self.hand_body_idxs, 3:7]
        self.init_vel = self.initial_rigid_body_states[self.hand_body_idxs, 7:]

        self.handle_pos = self.rigid_body_tensor[self.handle_body_idxs, :3].to(self.device)
        self.handle_rot = self.rigid_body_tensor[self.handle_body_idxs, 3:7]

        self.dof_dim = self.franka_num_dofs + 1
        self.pos_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.eff_act = torch.zeros((self.num_envs, self.dof_dim), device=self.device)
        self.y_dir = torch.Tensor([1, 0, 0]).to(self.device).view(1, 3)
        self.end_sim_flag = False
    
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)  #在base_task中

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)  #父类，这里创建了self.sim
        self._create_ground_plane()
        self._place_agents(self.cfg["env"]["numTrain"]+self.cfg["env"]["numVal"], self.cfg["env"]["envSpacing"])
    
    def control_ik(self,dpose):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (self.damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
    
    ##########################训练############################
    def step(self, actions):
        '''
        actions: 相当于score map 每个点的得分
        '''
        # print("actions.shape: ", actions.shape)  #5,4,4 tensor cuda
       
        self.grasp_times = 200
        per_step_force = self._move_ee(actions)  #TODO 考虑一下如果k无法整除num_envs的情况
        
        return per_step_force
        
        
    def reset(self, to_reset = "all"):

        self._partial_reset(to_reset)  # reset_buf 设为0

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless :
            self.render()
        if self.cfg["env"]["enableCameraSensors"] == True:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True) #new
        
        self.gym.render_all_camera_sensors(self.sim)  #渲染相机图像 527新增
        
        self.grasp_poses, grasp_scores, self.agnostic_high_idx, self.sample_points = self.calculate_grasp() #reset的重点
        
        # self._refresh_observation()      #更新self.obs_buf, 把采样点放入其中, 这里思考一下还需要obs buf不
        
        # success = self.success.clone()
        # reward, done = self._get_reward_done()  #获得self.reset_buf, rew_buf, 在step里也会执行, done应该是self.reset_buf的值, 为1

        return self.grasp_poses, self.agnostic_high_idx, self.sample_points #这里在ppt里面刚好对应

    def _partial_reset(self, to_reset = "all"):

        """
        reset those need to be reseted
        """

        if to_reset == "all" :
            to_reset = np.ones((self.env_num,))
        reseted = False
        for env_id, reset in enumerate(to_reset):
            if reset.item() :
                # need randomization
                reset_dof_states = self.initial_dof_states[env_id].clone()
                reset_root_states = self.initial_root_states[env_id].clone()
                
                self.dof_state_tensor[env_id].copy_(reset_dof_states)
                self.root_tensor[env_id].copy_(reset_root_states)
                reseted = True
                self.progress_buf[env_id] = 0
                self.reset_buf[env_id] = 0     #这里把resetbuf设置成0
                # self.success_buf[env_id] = 0  #这里success可以用来计算门把手下压角度
    
        if reseted:
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state_tensor))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_tensor))
            
    
    def _refresh_observation(self):
        '''
        self.sampled_pc
        求self.obs_buf  [num_envs, num_points, 3+dim]
        '''
        #刷新 actor
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim) #new
        self.gym.refresh_jacobian_tensors(self.sim)
        
        
                
        # print(self.obs_buf.shape)  #2 55
        # print(self.sample_points.shape) #1024 3
        
        # self.obs_buf[:, :, :3].copy_(self.sample_points)   #这里用点云填充了前3
        # self.obs_buf[:, :, 3:] = self._get_base_observation().view(self.num_envs, 1, -1).repeat_interleave(self.pointCloudDownsampleNum, dim=1)
        
        
    def _get_base_observation(self):
        '''
        num_env, dim
        '''
        
        hand_rot = self.hand_rigid_body_tensor[..., 3:7]
        hand_down_dir = quat_axis(hand_rot, 2)
        self.hand_tip_pos = self.hand_rigid_body_tensor[..., 0:3] + hand_down_dir * self.gripper_length    #夹爪的中间点
        self.hand_rot = hand_rot

        dim = 6 #

        state = torch.zeros((self.num_envs, dim), device=self.device) #
        
        state[:, :2].copy_(self.sensor_torques[..., 0])  #这里对力进行一次预处理，把reward先放小
        state[:, 2:4].copy_(self.sensor_torques[..., 1])  #这里对力进行一次预处理，把reward先放小
        state[:, 4:6].copy_(self.sensor_torques[..., 2])  #这里对力进行一次预处理，把reward先放小
        
        return state
    
    def _get_reward_done(self):
        # 这里计算reward
        # 返回奖励和是否重置
        # 几乎所有的存放信息前面都是env，后面才是实际的数值
        force_reward = (torch.sqrt(self.l_f[0] ** 2 + self.l_f[1] ** 2 + self.l_f[2] ** 2) + \
                                torch.sqrt(self.r_f[0] ** 2 + self.r_f[1] ** 2 + self.r_f[2] ** 2)) / 50
        # print("------------force_reward----------", force_reward)
        force_reward = 1.0 / (1.0 + force_reward**2)
        force_reward *= force_reward
        
        self.rew_buf = 1.0 * force_reward #减去open_reward的作用是
        # self.reset_buf = torch.ones((self.num_envs,), device=self.device)
        return self.rew_buf, self.reset_buf
        
    ########################抓取姿态计算########################
    def calculate_grasp(self):
        if not os.path.exists("graphics_images"):
            os.mkdir("graphics_images")
            
        depth_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.env_ptr_list[0], self.camera_handles_list[0], gymapi.IMAGE_DEPTH)
        #彩色图像
        color_image = self.gym.get_camera_image_gpu_tensor(self.sim, self.env_ptr_list[0], self.camera_handles_list[0], gymapi.IMAGE_COLOR)
        
        depth_image = gymtorch.wrap_tensor(depth_image)
        color_image = gymtorch.wrap_tensor(color_image)
        
        # -inf implies no depth value, set it to zero. output will be black.
        depth_image[depth_image == -torch.inf] = 0

        # clamp depth image to 10 meters to make output image human friendly
        depth_image[depth_image < -10] = -10
        
        # print('-'*20, 'color_image', color_image.shape)  (640, 480, 4)
        # print(color_image)
        rgb_np = color_image.cpu().numpy()[:, :, :3]
        # print(rgb_np.shape) #(640, 480, 3)
        bgr_np = rgb_np[..., [2,1,0]]
        depth_np = depth_image.cpu().numpy()
        
        # crop 可视化
        # WINDOW_NAME = "Grasp detections"
        # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # cv2.imshow(WINDOW_NAME, bgr_np)  #可视化需要BGR图像
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        grasp_pose_c, scores, high_idx, sample_points = pcfgrasp(bgr_np, depth_np, self.camera_view_matrix_list[0].cpu(), self.camera_proj_matrix_list[0].cpu(),
                            self.camera_u2.cpu(), self.camera_v2.cpu(), self.camera_properties.width, self.camera_properties.height, 10, self.device)
        # numpy 1x4x4, 1, 1, 1x3
        return grasp_pose_c, scores, high_idx, sample_points

    def _move_ee(self, grasp_pose_c):
        '''
        这里存储一个force buffer 用来后面计算reward
        移动夹爪抓紧把手并移动
        grasp_pose_c: {np.array} Nx4x4
        '''
        # print("grasp_pose_c", grasp_pose_c)
        dof_pos = self.dof_state_tensor[:, 0].view(self.env_num,self.env_dof_num,1).to(self.device)
        dof_vel = self.dof_state_tensor[:, 1].view(self.env_num,self.env_dof_num,1)
        pos_action = torch.zeros_like(dof_pos).squeeze(-1)
        # print(self.num_envs)
        
        down_flag = torch.full([self.num_envs], False, dtype=torch.bool).to(self.device).squeeze(-1).view(5,1)
        
        return_to_start = torch.full([self.num_envs], False, dtype=torch.bool).to(self.device)
        hand_restart = torch.full([self.num_envs], False, dtype=torch.bool).to(self.device)
        
        
        while(not self.end_sim_flag):
        
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim) 
            self.gym.refresh_mass_matrix_tensors(self.sim)
            
            
            grasp_pose_w = self.cam2world(grasp_pose_c).to(self.device)  # N, 4, 4
            
            target_rot = grasp_pose_w[:, :3, :3] # N, 3, 3
            
            target_pos = grasp_pose_w[:, 3, :3] # N, 3
            target_quat = torch.Tensor(rot2quat(target_rot)).to(self.device) # N, 4
            
            
            #self.hand_rigid_body_index
            hand_pos = self.rb_states[self.hand_body_idxs, :3]
            hand_rot = self.rb_states[self.hand_body_idxs, 3:7]
            hand_vel = self.rb_states[self.hand_body_idxs, 7:]
            
            grasp_offset = 0.11
            handle_radius = 0.009
            
            # print(target_pos)
            # print(hand_pos)
            
            to_handle = target_pos - hand_pos
            handle_dist = torch.norm(to_handle, dim=-1).unsqueeze(-1)
            handle_dir = to_handle / handle_dist
            handle_dot = handle_dir @ self.y_dir.view(3, 1)
            
            box_yaw_dir = quat_axis(self.handle_rot, 0).to(self.device)
            hand_yaw_dir = quat_axis(hand_rot, 0)
            yaw_dot = torch.bmm(box_yaw_dir.view(self.num_envs, 1, 3), hand_yaw_dir.view(self.num_envs, 3, 1)).squeeze(-1)
            
            gripper_sep = dof_pos[:, 7] + dof_pos[:, 8]
            gripped = (gripper_sep < 2*0.009) & (handle_dist < grasp_offset + 0.009)
            
            
            to_init = self.init_pos - hand_pos
            init_dist = torch.norm(to_init, dim=-1)
            hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
            return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)
            
            
            downside_pose = target_pos.clone().to(self.device)
            downside_pose[:,2] -= 0.02
            downside_quat = target_quat
    

            above_handle = ((handle_dot >= 0.99) & (yaw_dot >= 0.95) & (handle_dist < grasp_offset * 3)).squeeze(-1)
            grasp_pos = target_pos.clone()
            grasp_pos[:, 2] = torch.where(above_handle, target_pos[:, 0] + grasp_offset, target_pos[:, 0] + grasp_offset * 2.5)
            
            
            down_flag = gripped | down_flag
            
            down_flag = down_flag & (self.handle_pos[:, 2] > -0.01).unsqueeze(-1)
            
            goal_pos = torch.where(down_flag, grasp_pos, downside_pose)
            goal_rot = torch.where(down_flag, target_quat, downside_quat)
            
            goal_pos = torch.where(return_to_start, goal_pos, self.init_pos)
            goal_rot = torch.where(return_to_start, goal_rot, self.init_rot)
            
            pos_err = goal_pos - hand_pos
            orn_err = orientation_error(goal_rot, hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
            pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
            
            close_gripper = (handle_dist < grasp_offset + 0.008) | gripped
            
            # print(self.sensor_forces)
            # always open the gripper above a certain height, dropping the box and restarting from the beginning
            hand_restart = hand_restart | (self.handle_pos[:, 2] < -0.01)
            keep_going = torch.logical_not(hand_restart)
            close_gripper = close_gripper & keep_going.unsqueeze(-1)
            
            
            grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), torch.Tensor([[0.015, 0.015]] * self.num_envs).to(self.device))
            pos_action[:, 7:9] = grip_acts
            
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action.to('cpu')))

            self.end_sim_flag = torch.all(init_dist < 0.01)
            
            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)
        
        

    def cam2world(self, grasp_pose_c):
        '''
        Turn the grasp pose from virtual camera to panda robot
        
        grasp_pose_c {N 4x4 tensor}: camera grasp poses
        
        return:
            grasp_pose_w {N 4x4 tensor}: world grasp poses
        '''
        # print(grasp_pose_c.shape) #(1, 4, 4
        N, _, _ = grasp_pose_c.shape
        grasp_pose_w = torch.zeros((N, 4, 4), device=self.device)
        
        T_wc = torch.zeros((4, 4), device=self.device)
        
        for i in range(N):
            
            T_wc = T_wc.copy_(self.camera_view_matrix_list[0]).to(self.device)
            
            coor_R_1 = torch.asarray([[1.0, 0.0, 0.0, 0.0],
                                [0.0, -1.0, 0.0, 0.0],
                                [0.0, 0.0, -1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]], device=self.device)
            
            T_cw = torch.linalg.inv(T_wc)
            T_cw = torch.matmul(T_cw, coor_R_1).to(self.device)
            
            grasp_pose_w[i] = torch.matmul(T_cw, grasp_pose_c[i])
        
        # print('grasp_pose_w', grasp_pose_w.shape)
        return grasp_pose_w
        
    ########################load object#######################
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 0.1
        plane_params.dynamic_friction = 0.1
        self.gym.add_ground(self.sim, plane_params)
        
    def _place_agents(self, env_num, spacing):

        print("Simulator: creating agents")

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        self.space_middle = torch.zeros((env_num, 3), device=self.device)
        self.space_range = torch.zeros((env_num, 3), device=self.device)
        self.space_middle[:, 0] = self.space_middle[:, 1] = 0
        self.space_middle[:, 2] = spacing/2
        self.space_range[:, 0] = self.space_range[:, 1] = spacing
        self.space_middle[:, 2] = spacing/2
        num_per_row = int(np.sqrt(env_num))

        with tqdm(total=env_num) as pbar:
            pbar.set_description('Enumerating envs:')
            for env_id in range(env_num) :
                env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
                self.env_ptr_list.append(env_ptr)
                self._load_franka(env_ptr, env_id)
                self._load_door(env_ptr, env_id)
                self._load_camera(env_ptr, env_id)
                pbar.update(1)
    
    def _load_franka(self, env_ptr, env_id):
        
        # 因为是并行环境，所以只要加载一次模型，类就记住了，后面就不用再每次都加载assets
        if self.franka_loaded == False:
            self.franka_actor_list = []
            
            asset_root = self.asset_root
            asset_file = "franka_description/robots/franka_panda.urdf"
            self.gripper_length = 0.11
            
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            
            asset_options.flip_visual_attachments = True
            asset_options.armature = 0.01
            self.franka_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
            
            #创建夹爪的force sensor  这里是我自己加的
            franka_bodies = self.gym.get_asset_rigid_body_count(self.franka_asset)
            body_names = [self.gym.get_asset_rigid_body_name(self.franka_asset, i) for i in range(franka_bodies)]
            
            left_gripper_names = [name for name in body_names if 'panda_leftfinger' in name]
            right_gripper_names = [name for name in body_names if 'panda_rightfinger' in name]
                       
            left_gripper_index = [self.gym.find_asset_rigid_body_index(self.franka_asset, name) for name in left_gripper_names]
            right_gripper_index = [self.gym.find_asset_rigid_body_index(self.franka_asset, name) for name in right_gripper_names]
            
            sensor_pose = gymapi.Transform()
            for idx in left_gripper_index:
                self.gym.create_asset_force_sensor(self.franka_asset, idx, sensor_pose)
            for idx in right_gripper_index:
                self.gym.create_asset_force_sensor(self.franka_asset, idx, sensor_pose)
                
            self.franka_loaded = True
        
        #这里将关节信息转换成tensor
        franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits = self._get_dof_property(self.franka_asset)
        self.franka_dof_max_torque_tensor = torch.tensor(franka_dof_max_torque, device=self.device)
        self.franka_dof_mean_limits_tensor = torch.tensor((franka_dof_lower_limits + franka_dof_upper_limits)/2, device=self.device)
        self.franka_dof_limits_range_tensor = torch.tensor((franka_dof_upper_limits - franka_dof_lower_limits)/2, device=self.device)
        self.franka_dof_lower_limits_tensor = torch.tensor(franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits_tensor = torch.tensor(franka_dof_upper_limits, device=self.device)
        
        # 这里又捕获了上述的关节信息
        dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        
        #使用关节 position drive
        if self.cfg["env"]["driveMode"] in ["pos", "ik"]:   #关节位置驱动
            dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_POS)
            dof_props["stiffness"][:-2].fill(400.0)
            dof_props["damping"][:-2].fill(40.0)
        else:  # osc  驱动力驱动
            dof_props["driveMode"][:-2].fill(gymapi.DOF_MODE_EFFORT)
            dof_props["stiffness"][:-2].fill(0.0)
            dof_props["damping"][:-2].fill(0.0)
        
        # grippers  夹爪的驱动方式
        dof_props["driveMode"][-2:].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"][-2:].fill(400.0)
        dof_props["damping"][-2:].fill(80.0)
        
        #位置初始化
        initial_franka_pose = self._franka_init_pose()
        
        # set start dof
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset) #11
        default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        default_dof_pos[:-2] = (franka_dof_lower_limits + franka_dof_upper_limits)[:-2] * 0.3  #关节的位置
        
        # gripper open
        default_dof_pos[-2:] = franka_dof_upper_limits[-2:] #在这里设置了gripper的初始位置
        franka_dof_state = np.zeros_like(franka_dof_max_torque, gymapi.DofState.dtype) #设置了gripper的初始力矩，franka_dof_state是一个结构体
        franka_dof_state["pos"] = default_dof_pos #设置了gripper的初始位置
        
        # 创建actor
        franka_actor = self.gym.create_actor(env_ptr, self.franka_asset, initial_franka_pose, "franka", env_id, 2, 0)
        
        self.gym.set_actor_dof_properties(env_ptr, franka_actor, dof_props)
        self.gym.set_actor_dof_states(env_ptr, franka_actor, franka_dof_state, gymapi.STATE_ALL)
        self.franka_actor_list.append(franka_actor)
        
        
    def _load_door(self, env_ptr, env_id):
        
        if self.door_loaded == False:
            self.door_actor_list = []
        
            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True
            asset_options.collapse_fixed_joints = True
            asset_options.use_mesh_materials = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 512
            
            self.door_asset = self.gym.load_asset(self.sim, self.asset_root, 'model_door2/urdf/model_door2.urdf', asset_options)
            
            # #TODO 初始化位姿可以考虑单独写一个函数
            # self.initial_door_pose = gymapi.Transform()
        
            # self.initial_door_pose.r = gymapi.Quat(0.0, 0.0, 0.7071, 0.7071) #xyzw
            # self.initial_door_pose.p = gymapi.Vec3(-0.10, -0.15, 0.5)
            
            self.door_loaded = True
        
        rig_dict = self.gym.get_asset_rigid_body_dict(self.door_asset)
        # print(rig_dict)
        assert(len(rig_dict) == 2)
        self.door_rig_name = list(rig_dict.keys())[1]        #这里是门把手
        self.door_base_rig_name = list(rig_dict.keys())[0]   #这里应该是门
        
        dof_dict = self.gym.get_asset_dof_dict(self.door_asset)
        # print(dof_dict) {'joint1': 0}
        #这里有点懵
        if len(dof_dict) != 1:
            print(len(dof_dict))
        assert(len(dof_dict) == 1)
        self.door_dof_name = list(dof_dict.keys())[0]
        
        initial_door_pose = self._obj_init_pose()
        
        # franka_dof_max_torque, franka_dof_lower_limits, franka_dof_upper_limits = self._get_dof_property(self.door_asset)
        door_actor = self.gym.create_actor(env_ptr, self.door_asset, initial_door_pose, "door", env_id, 2, 0)
            
        self.door_actor_list.append(door_actor)
        
    
    def _load_camera(self, env_ptr, env_id):
        '''
        考虑到相机只生成一次,且在多个环境中同时使用,所以用self
        '''
        
        if self.camera_loaded == False:
            
            self.camera_handles_list = []
            self.camera_view_matrix_list = []
            self.camera_proj_matrix_list = []
        
            self.camera_loaded = True
        
        self.camera_u = torch.arange(0, self.camera_properties.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_properties.height, device=self.device)
        
        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')
        
        # print('-'*10, env_id, '-'*10)
        camera = self.gym.create_camera_sensor(env_ptr, self.camera_properties)
        # print('-'*10, camera, '-'*10)  0
        
        camera_position = gymapi.Vec3(0.5, 0.3, 1.5)
        camera_target = gymapi.Vec3(0, 0, 1.5)
        # print('-'*10, 'baocuojiance', '-'*10)
        self.gym.set_camera_location(camera, env_ptr, camera_position, camera_target)
        # print('-'*10, 'baocuojiance', '-'*10)
        
        cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera)))).to(self.device)
        cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera), device=self.device)
        
        self.camera_handles_list.append(camera)
        self.camera_view_matrix_list.append(cam_vinv)
        self.camera_proj_matrix_list.append(cam_proj)
    
    
    ########################参数设定###########################
    def _get_dof_property(self, asset):
        # 仅仅从gym环境中获得到了关节信息，然后把其转换成np矩阵
        # 作用是将关节信息在后续转换成tensor
        dof_props = self.gym.get_asset_dof_properties(asset)
        dof_num = self.gym.get_asset_dof_count(asset)
        dof_lower_limits = []
        dof_upper_limits = []
        dof_max_torque = []
        
        for i in range(dof_num):
            dof_max_torque.append(dof_props['effort'][i])
            dof_lower_limits.append(dof_props['lower'][i])
            dof_upper_limits.append(dof_props['upper'][i])
        dof_max_torque = np.array(dof_max_torque)
        dof_lower_limits = np.array(dof_lower_limits)
        dof_upper_limits = np.array(dof_upper_limits)
        return dof_max_torque, dof_lower_limits, dof_upper_limits
    
    def _franka_init_pose(self):
        initial_franka_pose = gymapi.Transform()
        
        initial_franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)
        initial_franka_pose.p = gymapi.Vec3(0.5, 0.0, 0.7)

        return initial_franka_pose
    
    def _obj_init_pose(self):
        #根据任务设定不同的位置
        initial_door_pose = gymapi.Transform()
        
        initial_door_pose.r = gymapi.Quat(0.0, 0.0, 0.7071, 0.7071) #xyzw
        initial_door_pose.p = gymapi.Vec3(-0.10, -0.15, 0.5)
        
        return initial_door_pose