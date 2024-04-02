from gym.spaces import Space, Box
import torch
from tqdm import tqdm
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
# from agents.ppo import RolloutStorage
import torch.nn as nn
import os
import numpy as np
import statistics
from run_utils.register import AGENTS
from run_utils.logger import get_root_logger, print_log
from agents.task_score_model import TaskScoreLoss, TaskScoreModel
from utils.force_buffer import ForceBuffer
from utils.visual_point import vis_score_pc, vis_pc
import matplotlib.pyplot as plt
from task_dataloader import TaskDataset
from torch.utils.data import DataLoader
from utils.covert_pc import center_pc_convert_cam
# from torchsummary import summary
# from thop import profile, clever_format


@AGENTS.register('real_time_sim')
class RealTimeSimAgent:
    
    def __init__(self, 
                 vec_env, 
                 num_mini_batches,
                 policy_cfg=None,
                 device='cpu',
                 log_dir='run',
                 log_subname=None,
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 ):
        #Env
        self.vec_env = vec_env
        
        # Model
        self.model = TaskScoreModel(policy_cfg).to(device)
        
        # self.model.load_state_dict(torch.load('/home/cyf/task_grasp/ABCDEFG/pcfgrasp_method/checkpoints/train/03-07-08-05_best_train_94.pth'), strict=False)
        
        # parameters = self.model.parameters()
        # for param in parameters:
        #     print(param)
        
        # model FLOPs
        # summary(self.model, (1024, 3))
        # input_tensor = torch.randn((2,1024,3)).cuda()
        # flops, params = profile(self.model, inputs=(input_tensor,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print("FLOPs: %s" %(flops))
        # print("params: %s" %(params))
        # raise ValueError
    
        # print(self.model.state_dict())
        self.loss = TaskScoreLoss(policy_cfg).to(device)
        
        # Log
        self.log_dir = log_dir
        self.log_subname = log_subname
        self.print_log = print_log #True
        # self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        
        self.is_testing = is_testing
        self.current_learning_iteration = 0
        self.apply_reset = apply_reset  #False
        self.tot_timesteps = 0
        self.tot_time = 0
        
        self.apply_reset = apply_reset
        self.exp_name = self.vec_env.task.exp_name
        
        self.num_mini_batches = num_mini_batches #100

        
        self.log_step = self.num_mini_batches // 1
        # print('-'*10, 'log_step', self.log_step)
        
        ########cudnn
        # torch.backends.cudnn.benchmark = False
        
        ################optimizer################
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, betas=(0.9, 0.999))   #learning_rate
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.7)
        
        self.train_env_num = policy_cfg["training_env_num"]
        self.val_env_num = policy_cfg["valitating_env_num"]
        
        self.iteration_per_buffer = int(policy_cfg["buffer_size"] / policy_cfg["num_env"])  #一次生成的抓取结果需要多少个循环来填满一个buffer
        self.buffer_num_per_batch = policy_cfg["batch_group_length"]   #这里可以保存好几组点云成为一个batch来进行训练
        self.max_epoch = policy_cfg["max_epoch"] #这里和上面的都是新增的
        
        self.train_num_mini_batches = self.num_mini_batches #100
        self.eval_num_mini_batches = 1 #10 TODO 设定成自选的
        
##################################################################
        
        self.device = device
        
        #model config，在点云的时候用到
        self.num_sample_points = policy_cfg["num_sample_points"]  #1024
        
        self.policy_cfg = policy_cfg
        
        # PPO components
        self.vec_env = vec_env
        
        self.force_buffer = ForceBuffer(buffer_size=policy_cfg["buffer_size"],
                                        content_dim=1,
                                        task_top_k = policy_cfg["topk_confidence"],
                                        device=self.device)
                
    
    def load(self, path):
        '''
        '''
        self.model.load_state_dict(torch.load(path, map_location='cuda'))
        
    
    def inference(self, path):
        '''
        推理
        '''
        self.load(path)
        
        self.model.eval()
        with torch.no_grad():
            
            for epoch in range(self.max_epoch): #eval的时候只有一个batch
                grasp_poses, agnostic_high_idx, sample_points = self.vec_env.reset()
                
                batch_sample_points = sample_points.unsqueeze(0)

                # ---------   平面点云
                batch_points_all = generate_plane_point_cloud(1024, 0.2)
                batch_sample_points = batch_points_all.unsqueeze(0).cuda()
                # ---------
                
                batch_sample_points = center_pc_convert_cam(batch_sample_points)
                
                # batch_sample_points = center_pc_convert_cam(batch_sample_points)
                
                task_scores = self.model(batch_sample_points)    #B, N, 1   #[:, :, [1, 0, 2]]
                print('task_scores', torch.max(task_scores))
                print('task_scores', task_scores)
                
                # top_k_scores, top_k_idx = torch.topk(task_scores, 1, dim=1)
                
                # selected_grasp = grasp_poses[top_k_idx]
                # self.vis_score(task_scores.squeeze(0).cpu().numpy())
                
                # vis_score_pc(sample_points, task_scores)
                #------------
                vis_score_pc(batch_sample_points.squeeze(0).cpu().numpy(), task_scores)
    
    def vis_score(self, task_scores):
        score_list_x = []
        score_list_y = []
        
        for i in range(len(task_scores)):
            score_list_x.append(i)
            score_list_y.append(task_scores[i])
            
        plt.figure()
        plt.title('score')
        plt.scatter(score_list_x, score_list_y, color='red', marker='o')
        plt.show()
        
    def capture_pic_for_seg(self, pic_nums):
        '''
        用来生成用于分割的图片
        '''
        self.vec_env.capture_4_seg(pic_nums)
        
    
    def run(self, dataset_path=None):
        
        # 训练的时候加入logger
                
        # logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        self.log_file = os.path.join(self.log_dir, f'{timestamp}_{self.exp_name}_{self.log_subname}.log')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.logger = get_root_logger(self.log_file, name='log')
        
        if dataset_path is not None:
            self._dataset_train(dataset_path)
        else:
            self._sim_train()
            
    
    def save(self, path):
        #保存模型
        torch.save(self.model.state_dict(), path)
        print_log('Model Saved in file: %s' % path, logger=self.logger)

    
    def save_scene_sim(self, points, scores, num_epoch, num_batch):
        '''
        保存场景仿真的结果
        points: 点云 {tensor} Nx3
        scores: 力的大小 {tensor} BxNx3
        '''
        points = points.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()

        
        if not os.path.exists(os.path.join(self.log_dir, 'dataset')):
            os.makedirs(os.path.join(self.log_dir, 'dataset'))
            
        np.save(os.path.join(self.log_dir, 'dataset', 'points_{}_{}.npy'.format(num_epoch,num_batch)), points)
        np.save(os.path.join(self.log_dir, 'dataset', 'scores_{}_{}.npy'.format(num_epoch,num_batch)), scores)

        
    
    
    def _sim_train(self):
        
        # 这里就一个数，因为是vec env中的
        # current_states = self.vec_env.get_state()  #这个参数在训练中并未被传递, 因为states_buf在环境的文件夹中并未被定义或者传递信息
        
        tmp_loss = 1e6
        # batch_idx = torch.empty((self.buffer_num_per_batch, 50), device=self.device) #1
        
        for epoch in range(self.max_epoch):   #总的epoch
            
            self.model.train()
            
            for batch_idx in range(self.train_num_mini_batches): #每个epoch中的batch
                
                batch_score_label = torch.zeros(self.buffer_num_per_batch, self.num_sample_points, 1, device=self.device) #用来存放一个batch的buffer数据，避免每个batch只有一个buffer导致的训练不稳定
                batch_points = torch.zeros(self.buffer_num_per_batch, self.num_sample_points, 3, device=self.device) #3
                
                
                for buf_idx in range(self.buffer_num_per_batch):  #每个batch中的buffer，因为一个buffer是通过一次对环境的拍摄得到的
                    print(f'---------train{epoch}-{batch_idx}/{buf_idx}---------')
                    
                    grasp_poses, agnostic_high_idx, sample_points = self.vec_env.reset()  #相机拍摄生成抓取结果，task_agnostic得分，和对应的采样点 N 4 4, N 1, N 3
                    vis_pc(sample_points.detach().cpu().numpy())
                    # vis_pc(sample_points[agnostic_high_idx, ...].detach().cpu().numpy())
                    
                    for iter in range(self.iteration_per_buffer):
                        print(f'---------minibatch_train{iter}/{self.iteration_per_buffer}---------')
                        ##########0811 TODO
                        ##########
                        per_step_force = self.vec_env.step(grasp_poses[iter * self.policy_cfg["num_env"]: (iter + 1) * self.policy_cfg["num_env"], ...],
                                                           sample_points[iter * self.policy_cfg["num_env"]: (iter + 1) * self.policy_cfg["num_env"], ...])   # num_env, 1
                        # 这里的step是在环境中进行的，不是在模型中进行的
                        self.force_buffer.insert(per_step_force)

                        # print(per_step_force)
                    # print(self.force_buffer.all_label().shape)
                    # TODO 把force转换成对应的label
                    batch_score_label[buf_idx, agnostic_high_idx, :] = self.force_buffer.all_label()   #(Num_grasp, 1) 50 x 1 ####0811保留 0812
                    # batch_score_label[buf_idx, :, :] = self.force_buffer.all_label()   #(Num_grasp, 1) 50 x 1 #0812
                    print(self.force_buffer.all_label())
                    # buffer的第一维度，每一个，包含了一次环境仿真中所有的抓取的尝试
                    # vis_pc(sample_points[agnostic_high_idx, ...][force_idx].detach().cpu().numpy())
                    
                    batch_points[buf_idx, ...] = sample_points
                    # batch_idx[buf_idx, :] = agnostic_high_idx
                    self.save_scene_sim(sample_points, self.force_buffer.all_label(), epoch, buf_idx)
                    self.force_buffer.buffer_reset()
                    # force_idx = (self.force_buffer.buffer_reset()).squeeze(-1)         #因为在buffer中是一循环一填充，有一个top函数在帮忙控制填充位置，所以这里要重置一下
                    
                # vis_score_pc(batch_points[0], batch_score_label[0])
                # vis_score_pc(batch_points[1], batch_score_label[1])
                # vis_score_pc(batch_points[2], batch_score_label[2])
                # vis_score_pc(batch_points[3], batch_score_label[3])
                # vis_score_pc(batch_points[4], batch_score_label[4])
                
                batch_points = center_pc_convert_cam(batch_points) #这里相机的姿态在不同的batch中保持一致
                
                # print(batch_score_label.shape, batch_points.shape)  #torch.Size([1, 1024, 1]) torch.Size([1, 1024, 3])
                task_scores = self.model(batch_points)
                # vis_score_pc(batch_points[0], task_scores[0])
                # print(task_scores.shape)  #torch.Size([1, 1024, 1])
                task_score_loss = self.loss(task_scores, batch_score_label)
                task_score_loss.backward()

                
                if (batch_idx + 1) % self.log_step == 0:
                    print_log('Epoch: [{0}][{1}/{2}] mse_score_loss:{3}'.format(epoch, batch_idx + 1, self.num_mini_batches, 
                                                                                task_score_loss.detach().cpu().numpy()), logger=self.logger)
            self.optimizer.step()
            self.lr_scheduler.step()        
            
            
            self.model.eval()
            
            eval_total_loss = 0.
            with torch.no_grad():
                for batch_idx in range(self.eval_num_mini_batches): #每个epoch中的batch
                    
                    batch_score_label = torch.zeros(self.buffer_num_per_batch, self.num_sample_points, 1, device=self.device) #用来存放一个batch的buffer数据，避免每个batch只有一个buffer导致的训练不稳定
                    batch_points = torch.zeros(self.buffer_num_per_batch, self.num_sample_points, 3, device=self.device) #3
                    
                    for buf_idx in range(self.buffer_num_per_batch):  #每个batch中的buffer，因为一个buffer是通过一次对环境的拍摄得到的
                        print(f'---------eval{epoch}-{batch_idx}/{buf_idx}---------')
                        
                        grasp_poses, agnostic_high_idx, sample_points = self.vec_env.reset()  #相机拍摄生成抓取结果，task_agnostic得分，和对应的采样点
                        # vis_pc(sample_points[agnostic_high_idx, ...].detach().cpu().numpy())
                        for iter in range(self.iteration_per_buffer):
                            print(f'---------minibatch_train{iter}/{self.iteration_per_buffer}---------')

                            ############TODO 0811
                            per_step_force = self.vec_env.step(grasp_poses[iter * self.policy_cfg["num_env"]: (iter + 1) * self.policy_cfg["num_env"], ...],
                                                               sample_points[iter * self.policy_cfg["num_env"]: (iter + 1) * self.policy_cfg["num_env"], ...])   # num_env, 1
                            #这里的step是在环境中进行的，不是在模型中进行的
                            
                            self.force_buffer.insert(per_step_force)
                        
                        # TODO 把force转换成对应的label
                        batch_score_label[buf_idx, agnostic_high_idx, :] = self.force_buffer.all_label()   #B N 1
                        # batch_score_label[buf_idx, :, :] = self.force_buffer.all_label()   #(Num_grasp, 1) 50 x 1
                        
                        # buffer的第一维度，每一个，包含了一次环境仿真中所有的抓取的尝试
                        # print(batch_score_label[buf_idx, agnostic_high_idx, :])
                        self.force_buffer.buffer_reset()          #因为在buffer中是一循环一填充，有一个top函数在帮忙控制填充位置，所以这里要重置一下
                        batch_points[buf_idx, ...] = sample_points  #B N 3
                    
                    batch_points = center_pc_convert_cam(batch_points)
                    
                    task_scores = self.model(batch_points)
                    
                    task_score_loss = self.loss(task_scores, batch_score_label)          #因为在buffer中是一循环一填充，有一个top函数在帮忙控制填充位置，所以这里要重置一下
                    
                    eval_total_loss += task_score_loss.detach().cpu().numpy()
                    
            ###########################################################################
            avg_eval_loss = eval_total_loss / self.eval_num_mini_batches
            print_log('Epoch: [{0}] avg_eval_loss:{1}'.format(epoch, avg_eval_loss), logger=self.logger)
            
    
            
            if avg_eval_loss<tmp_loss:
                self.save(os.path.join(self.log_dir, 'model_{0}_epoch{1}.pt'.format(time.strftime("%m-%d-%H"), epoch)))
                tmp_loss = avg_eval_loss
    
    def _dataset_train(self, dataset_path):
        
        #TODO replace dataset_path with argparser
        train_dataset = TaskDataset(dataset_path=dataset_path, scene_num=100, scene_batch=3)
        #TODO replace batch_size with argparser
        train_dataloader = DataLoader(train_dataset, batch_size=self.buffer_num_per_batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        
        # print(len(train_dataloader))
        print('start train')
        # print(len(train_dataloader)) #500
        tmp_loss = 1e6
        for epoch in range(self.max_epoch):
            print(f'-------------{epoch}----------')
            
            self.model.train()
            
            
            for batch_idx, (batch_points, batch_force) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):
                # print(f'---------train{epoch}-{batch_idx}---------')
                # start_time = time.time()
                # print('-'*20)
                # print(batch_points.shape, batch_force.shape)
                #TODO if gpu
                batch_points = center_pc_convert_cam(batch_points)
                batch_points = batch_points.to(self.device)
                batch_score_label = batch_force.to(self.device)
                
                vis_score_pc(batch_points[0], batch_score_label[0])
                vis_score_pc(batch_points[1], batch_score_label[1])
                vis_score_pc(batch_points[2], batch_score_label[2])
                # vis_score_pc(batch_points[3], batch_score_label[3])
                # vis_score_pc(batch_points[4], batch_score_label[4])
                
                self.optimizer.zero_grad()
                # model
                task_scores = self.model(batch_points)
                # loss
                task_score_loss = self.loss(task_scores, batch_score_label)
                
                task_score_loss.backward()
                self.optimizer.step()
                
            self.lr_scheduler.step()
                # if (batch_idx + 1) % self.log_step == 0:
                #     print_log('Epoch: [{0}] mse_score_loss:{1}'.format(epoch, task_score_loss.detach().cpu().numpy()), logger=self.logger)
            
                 
            self.model.eval()
            
            eval_total_loss = 0.
            with torch.no_grad():
                for batch_idx, (batch_points, batch_force) in enumerate(tqdm(train_dataloader, total=len(train_dataloader), smoothing=0.9)):
                    
                    batch_points = center_pc_convert_cam(batch_points)
                    batch_points = batch_points.to(self.device)
                    batch_score_label = batch_force.to(self.device)
                    
                    # model
                    task_scores = self.model(batch_points)
                    # loss
                    task_score_loss = self.loss(task_scores, batch_score_label)
                    
                    eval_total_loss += task_score_loss.detach().cpu().numpy()
            ###########################################################################
            avg_eval_loss = eval_total_loss / self.eval_num_mini_batches
            # print('avg_eval_loss', avg_eval_loss)
            print_log('Epoch: [{0}] mse_score_loss:{1}'.format(epoch, avg_eval_loss), logger=self.logger)
            
            print('tmp_loss', tmp_loss)
            if avg_eval_loss<tmp_loss:
                self.save(os.path.join(self.log_dir, '{2}_model_{0}_epoch{1}.pt'.format(time.strftime("%m-%d-%H-%M"), epoch, self.log_subname)))
                tmp_loss = avg_eval_loss
                
                
def generate_plane_point_cloud(n, spacing=0.05):
    """
    生成坐标原点附近的平面点云
    Args:
        n (int): 点的数量
        spacing (float): 点的间距

    Returns:
        torch.Tensor: 生成的点云，shape为 (n, 3)
    """
    # 在 x 和 y 轴上生成均匀分布的点
    x_coords = torch.linspace(-spacing, spacing, int(torch.sqrt(torch.Tensor([n]))))
    y_coords = torch.linspace(-spacing, spacing, int(torch.sqrt(torch.Tensor([n]))))

    # 生成网格
    xx, yy = torch.meshgrid(x_coords, y_coords)

    # 将网格点转换为点云
    point_cloud = torch.stack((xx.flatten(), yy.flatten(), torch.zeros(n)), dim=1)

    return point_cloud.reshape(n, 3)

