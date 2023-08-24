import os
from isaacgym.torch_utils import *
from zmq import PROTOCOL_ERROR_ZAP_INVALID_STATUS_CODE
from tasks.franka_cabinet_PC_partial import OneFrankaCabinetPCPartial
from collision_predictor.train_with_cp import CollisionPredictor
from utils.gpu_mem_track import MemTracker
from run_utils.register import TASKS

@TASKS.register('OneFrankaCabinetPCPartialCPMap')
class OneFrankaCabinetPCPartialCPMap(OneFrankaCabinetPCPartial):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, log_dir=None):

        # print('-'*20, 'CPMap', '-'*20)
        self.CollisionPredictor = CollisionPredictor(cfg, log_dir)
        self.cabinet_mask_dim = 5
        
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index, is_multi_agent, log_dir=log_dir)

        self.num_obs += 1  #这里加1，
        self.task_meta["mask_dim"] = 3
        self.task_meta["obs_dim"] = self.num_obs
        self.task_meta['need_update'] = True
        self.num_feature = cfg["env"]["pointFeatureDim"]
        self.CP_iter = cfg['cp']['CP_iter']
        self.obs_buf = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, self.num_obs), device=self.device, dtype=torch.float)
        # print('-'*20, self.obs_buf.shape, '-'*20)   #torch.Size([1, 2112, 61])

        self.depth_bar = self.cfg["env"]["depth_bar"]
        self.success_rate_bar = self.cfg["cp"]["success_rate_bar"]
        self.raw_map = torch.zeros((self.cabinet_num, self.cabinetPCOriginalNum), device=self.device)
        self.map = torch.zeros((self.cabinet_num, self.cabinetPCOriginalNum), device=self.device)
        # The tensors representing pointcloud and map are:
        # self.cabinet_pc   (obj*8192*5)
        # self.raw_map      (obj*8192)
        # self.map          (obj*8192)

    def quat_apply(self, a, b):
        shape = b.shape
        a = a.reshape(-1, 4)  # 4
        a_expand = a.expand(shape[0], 4)
        b = b.reshape(-1, 3)  # num_buffer*3
        xyz = a_expand[:, :3]   # 3
        t = xyz.cross(b, dim=-1) * 2
        return (b + a_expand[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)
    
    def _refresh_map(self) :
        print('-'*20, 'refresh map cp map line 46', '-'*20)  #这里就被执行了一次
        # print(self.selected_cabinet_pc.shape)  #torch.Size([1, 8192, 4])
        # predict all points at the begin
        with torch.no_grad():
            sampled_pc = self.sample_points(   #这里是用最远点采样，获得的2048个点
                self.selected_cabinet_pc,
                self.cabinetPCDownsampleNum,
                sample_method="furthest",
            )
            # print(sampled_pc.shape)  #torch.Size([1, 2048, 4])
            transformed_pc = self._get_transformed_pc(    #这里可以看到，后面的第四纬度，用来存储mask
                pc=sampled_pc[:, :, :3],
                mask=sampled_pc[:, :, 3]
            )
            # print(transformed_pc.shape)  #torch.Size([1, 2048, 3])
            stacked_pc = torch.cat(      #这里把mask和点云拼接起来，保存的mask的信息的目的是什么？
                (
                    transformed_pc,
                    sampled_pc[:, :, 3].view(self.cabinet_num, -1, 1)
                ),
                dim=2
            )
            # print(stacked_pc.shape) #torch.Size([1, 2048, 4])
             
            #这里pred_one_batch没有target，这里用来更新map
            self.map = self.CollisionPredictor.pred_one_batch(
                stacked_pc,
                self.success_rate,
                num_train=self.env_num_train
            ).to(self.device)
            # print(self.map.shape)  #torch.Size([1, 2048]) 这里就是输出的成绩
            
            self.map *= sampled_pc[:, :, 3]    #通过这里可以看出，mask相当于抓取的权重，在这里进行了更新，通过这里是相乘，以及之前输出的是1，0，所以这里应该是数据集中的可抓区域
            self.sampled_cabinet_pc[:, :, :4] = sampled_pc.repeat_interleave(self.env_per_cabinet, dim=0)   #
            self.sampled_cabinet_pc[:, :, 4] = self.map.repeat_interleave(self.env_per_cabinet, dim=0)  #这里五个纬度的最后一个纬度是成绩

    def save(self, path, iteration) :

        super().save(path, iteration)
        self.CollisionPredictor.save_checkpoint(os.path.join(path, "CP_{}.pt".format(iteration)))

        save_pc = self.compute_point_cloud_state(pc=self.sampled_cabinet_pc)
        torch.save(self._detailed_view(save_pc)[:, 0, ...], os.path.join(path, "map_{}.pt".format(iteration)))

        transformed_pc = self._get_transformed_pc()
        saving_raw_map = torch.cat((transformed_pc[:, :, :3], self.raw_map[:, :].view(self.cabinet_num, -1, 1)), dim=-1)
        torch.save(saving_raw_map, os.path.join(path, "rawmap_{}.pt".format(iteration)))
    
    def load(self, path, iteration) :
        
        cp_file = os.path.join(path, "CP_{}.pt".format(iteration))
        print("loading CP checkpoint", cp_file)
        super().load(path, iteration)
        self.CollisionPredictor.load_checkpoint(cp_file)
        print('-'*20, 'load cp map line 93', '-'*20)
        self._refresh_map()

    def _data_argumentation(self, pcd):
        pcd[:, :, :3] *= torch.rand((pcd.shape[0], 1, 3), device=self.device)*0.3 + 0.85
        return pcd
    
    def _get_transformed_pc(self, pc=None, mask=None):
        '''
        有的地方输入的是原始的点云，有的地方输入的是机械臂移动后的点云
        '''

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

    def update(self, iter):
        
        # print('-'*10, 'update franka cab PC partial cp map', '-'*10)
    
        CP_info = {}
        used_success_rate = self._detailed_view(self.success_rate).mean(dim=-1)
        
        #这里一直在训练，但不进入这里的判断，直到成功率达到阈值，这里设定的成功率是0.1
        if used_success_rate.mean() > self.success_rate_bar:
            # do training only when success rate is enough

            transformed_pc = self._get_transformed_pc()
            #get_map的作用是，计算点云的得分，这里get_map函数来自于PC_partial
            self.raw_map = self.get_map(self.selected_cabinet_pc[:, :, :3], self.selected_cabinet_pc[:, :, 3], self.contact_buffer_list)

            # stack them together to make resample easier
            stacked_pc_target = torch.cat(
                (
                    transformed_pc,
                    self.selected_cabinet_pc[:, :, 3].view(self.cabinet_num, -1, 1),
                    self.raw_map.view(self.cabinet_num, -1, 1)
                ),
                dim=2
            )

            # in training, sample a few points to train CP each epoch
            minibatch_size = self.cfg["cp"]["cp_minibatch_size"]
            for i in range(self.CP_iter):
                info_list = []
                #这里采集了最新的点，点采样也是父类中的函数
                sampled_pc_target = self.sample_points(
                    stacked_pc_target,
                    self.cabinetPCDownsampleNum,
                    sample_method="furthest"
                )
                sampled_pc_target = self._data_argumentation(sampled_pc_target)
                for cur_pc_target, cur_success_rate in zip(
                        torch.split(sampled_pc_target, minibatch_size),
                        torch.split(used_success_rate, minibatch_size)
                    ) :
                    '''
                    这里应该看一下minibatch的大小
                    '''
                    # 这里的cur_pc_target是采样的点云，cur_success_rate是成功率,数据的纬度大小是
                    # self._refresh_pointcloud_visualizer(cur_pc_target[0, :, :3], cur_pc_target[0, :, 3])
                    # 循环起来的时候，执行的是这里的pred
                    cur_map, cur_info = self.CollisionPredictor.pred_one_batch(
                        cur_pc_target[:, :, :4],
                        cur_success_rate,
                        target=cur_pc_target[:, :, 4],
                        num_train=self.cabinet_num_train
                    )
                    #这里的cur_info是字典，包含了loss
                    info_list.append(cur_info)
            self.CollisionPredictor.network_lr_scheduler.step()

            # collecting training info
            if self.CP_iter :
                for key in info_list[0] :
                    tmp = 0
                    for info in info_list :
                        tmp += info[key]
                    CP_info[key] = tmp/len(info_list)

            # 这里的refresh应该是预测,更新了self.map
            self._refresh_map()

        return CP_info
    
    def _get_max_point(self, pc, map) :

        env_max = map.max(dim=-1)[0]
        weight = torch.where(map > env_max.view(self.env_num, 1)-0.1, 1, 0)  # 选择了比其小0.1的点作为阈值
        weight_reshaped = weight.view(self.env_num, -1, 1)
        mean = (pc*weight_reshaped).mean(dim=1)
        return mean

    def _get_reward_done(self):
        '''
        这里多把夹爪到点云的距离算了进去
        '''

        rew, res = super()._get_reward_done()

        d = torch.norm(self.hand_tip_pos - self._get_max_point(self.sampled_cabinet_pc[:, :, :3], self.sampled_cabinet_pc[:, :, 4]), p=2, dim=-1)
        dist_reward = 1.0 / (1.0 + d**2)
        dist_reward *= dist_reward
        dist_reward = torch.where(d <= 0.1, dist_reward*2, dist_reward)
        rew += dist_reward * self.cfg['cp']['max_point_reward']

        return rew, res

    def _refresh_observation(self) :
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg["env"]["driveMode"] == "ik" :
            self.gym.refresh_jacobian_tensors(self.sim)

        # print('map obs refresh')
        # print('-'*10, 'refresh observation', '-'*10)
        # print('self.sampled_cabinet_pc', self.sampled_cabinet_pc.shape)  #([1, 2048, 5])
        point_clouds = self.compute_point_cloud_state(pc=self.sampled_cabinet_pc)
        # print('point_clouds', point_clouds.shape)  #([1, 2112, 6])   2048+64   最后一个纬度是用来区分手和柜子的

        point_clouds[:, :, :3] -= self.franka_root_tensor[:, :3].view(-1, 1, 3)   #这里的目的是为了把点云的坐标系转换到机械臂的坐标系下
        self.obs_buf[:, :, :6].copy_(point_clouds)
        if self.cfg["cp"]["max_point_observation"]:    #True
            self.obs_buf[:, :, 6:] = self._get_base_observation(self._get_max_point(point_clouds[:, :, :3], point_clouds[:, :, 4])).view(self.num_envs, 1, -1).repeat_interleave(self.pointCloudDownsampleNum, dim=1)
            # print(self.obs_buf.shape) torch.Size([1, 2112, 61])
        else :
            self.obs_buf[:, :, 6:] = self._get_base_observation().view(self.num_envs, 1, -1).repeat_interleave(self.pointCloudDownsampleNum, dim=1)
        if self.pointCloudVisualizer != None:
            self._refresh_pointcloud_visualizer(point_clouds[0, :, :3], point_clouds[0, :, 4])