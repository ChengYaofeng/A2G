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
import statistics
from run_utils.register import AGENTS
from run_utils.logger import get_root_logger, print_log
from agents.task_score_model import TaskScoreLoss, TaskScoreModel
from utils.force_buffer import ForceBuffer
from utils.visual_point import vis_score_pc, vis_pc
import matplotlib.pyplot as plt

@AGENTS.register('real_time_sim')
class RealTimeSimAgent:
    
    def __init__(self, 
                 vec_env, 
                 num_mini_batches,
                 model_cfg=None,
                 device='cpu',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 ):
        #Env
        self.vec_env = vec_env
        
        # Model
        self.model = TaskScoreModel(model_cfg).to(device)
        
        
        self.loss = TaskScoreLoss(model_cfg).to(device)
        
        # Log
        self.log_dir = log_dir
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
        
        # logger
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(self.log_dir, f'{timestamp}_{self.exp_name}.log') 
        self.logger = get_root_logger(log_file, name='log')
        
        self.log_step = self.num_mini_batches // 1
        print('-'*10, 'log_step', self.log_step)
        
        ########cudnn
        # torch.backends.cudnn.benchmark = False
        
        ################optimizer################
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))   #learning_rate
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.7)
        
        self.train_env_num = model_cfg["training_env_num"]
        self.val_env_num = model_cfg["valitating_env_num"]
        
        self.iteration_per_buffer = int(model_cfg["buffer_size"] / model_cfg["num_env"])  #一次生成的抓取结果需要多少个循环来填满一个buffer
        self.buffer_num_per_batch = model_cfg["batch_group_length"]   #这里可以保存好几组点云成为一个batch来进行训练
        self.max_epoch = model_cfg["max_epoch"] #这里和上面的都是新增的
        
        self.train_num_mini_batches = self.num_mini_batches #100
        self.eval_num_mini_batches = 1 #10 TODO 设定成自选的
        
##################################################################
        
        self.device = device
        
        #model config，在点云的时候用到
        self.num_sample_points = model_cfg["num_sample_points"]  #1024
        
        self.model_cfg = model_cfg
        
        # PPO components
        self.vec_env = vec_env
        
        self.force_buffer = ForceBuffer(buffer_size=model_cfg["buffer_size"],
                                        content_dim=1,
                                        task_top_k = model_cfg["topk_confidence"],
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
                # print('batch_sample_points', batch_sample_points.shape)
                task_scores = self.model(batch_sample_points)    #B, N, 1
                print('task_scores', torch.max(task_scores))
                
                # top_k_scores, top_k_idx = torch.topk(task_scores, 1, dim=1)
                
                # selected_grasp = grasp_poses[top_k_idx]
                # self.vis_score(task_scores.squeeze(0).cpu().numpy())
                
                vis_score_pc(sample_points, task_scores)
    
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
    
    def run(self):
        
        # 这里就一个数，因为是vec env中的
        current_states = self.vec_env.get_state()  #这个参数在训练中并未被传递, 因为states_buf在环境的文件夹中并未被定义或者传递信息
        
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
                    # print(grasp_poses, agnostic_high_idx, sample_points)
                    # vis_pc(sample_points.detach().cpu().numpy())
                    
                    # vis_pc(sample_points[agnostic_high_idx, ...].detach().cpu().numpy())
                    
                    # print(type(agnostic_high_idx))
                    for iter in range(self.iteration_per_buffer):
                        # print(f'---------minibatch_train{iter}/{self.iteration_per_buffer}---------')
                        per_step_force = self.vec_env.step(grasp_poses[iter * self.model_cfg["num_env"]: (iter + 1) * self.model_cfg["num_env"], ...])   # num_env, 1
                        #这里的step是在环境中进行的，不是在模型中进行的
                        # print(per_step_force.shape)
                        
                        self.force_buffer.insert(per_step_force)

                    # TODO 把force转换成对应的label
                    batch_score_label[buf_idx, agnostic_high_idx, :] = self.force_buffer.all_label()   #(Num_grasp, 1) 50 x 1
                    # buffer的第一维度，每一个，包含了一次环境仿真中所有的抓取的尝试
                    # print(batch_score_label[buf_idx, agnostic_high_idx, :])
                    force_idx = (self.force_buffer.buffer_reset()).squeeze(-1)         #因为在buffer中是一循环一填充，有一个top函数在帮忙控制填充位置，所以这里要重置一下
                    print(force_idx.shape)
                    # vis_pc(sample_points[agnostic_high_idx, ...][force_idx].detach().cpu().numpy())
                    
                    batch_points[buf_idx, ...] = sample_points
                    # batch_idx[buf_idx, :] = agnostic_high_idx
                    
                # print(batch_score_label.shape, batch_points.shape)  #torch.Size([1, 1024, 1]) torch.Size([1, 1024, 3])
                # print(batch_score_label)
                task_scores = self.model(batch_points)
                # print(task_scores)
                # print(task_scores.shape)  #torch.Size([1, 1024, 1])
                # print(torch.sum(batch_score_label, dim=1))
                # print(batch_score_label.shape)
                
                task_score_loss = self.loss(task_scores, batch_score_label, agnostic_high_idx)
                task_score_loss.backward()
                self.optimizer.step()
                
                if (batch_idx + 1) % self.log_step == 0:
                    print_log('Epoch: [{0}][{1}/{2}] bce_score_loss:{3}'.format(epoch, batch_idx + 1, self.num_mini_batches, 
                                                                                task_score_loss.detach().cpu().numpy()), logger=self.logger)
            
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
                            # print(f'---------minibatch_train{iter}/{self.iteration_per_buffer}---------')
                        
                            per_step_force = self.vec_env.step(grasp_poses[iter * self.model_cfg["num_env"]: (iter + 1) * self.model_cfg["num_env"], ...])   # num_env, 1
                            #这里的step是在环境中进行的，不是在模型中进行的
                            # print(per_step_force)
                            self.force_buffer.insert(per_step_force)
                        
                        # TODO 把force转换成对应的label
                        batch_score_label[buf_idx, agnostic_high_idx, :] = self.force_buffer.all_label()   #(Num_grasp, 1) 50 x 1
                        # buffer的第一维度，每一个，包含了一次环境仿真中所有的抓取的尝试
                        # print(batch_score_label[buf_idx, agnostic_high_idx, :])
                        self.force_buffer.buffer_reset()          #因为在buffer中是一循环一填充，有一个top函数在帮忙控制填充位置，所以这里要重置一下
                        batch_points[buf_idx, ...] = sample_points
                        # batch_idx[buf_idx, :] = agnostic_high_idx
                        
                        
                    # print(batch_score_label.shape, batch_points.shape)  #torch.Size([1, 1024, 1]) torch.Size([1, 1024, 3])
                    # print(batch_score_label)
                    task_scores = self.model(batch_points)
                    print(task_scores)
                    # print(torch.sum(batch_score_label, dim=1))
                    
                    # print(task_scores.shape)  #torch.Size([1, 1024, 1])
                    
                    task_score_loss = self.loss(task_scores, batch_score_label, agnostic_high_idx)          #因为在buffer中是一循环一填充，有一个top函数在帮忙控制填充位置，所以这里要重置一下
                    
                    eval_total_loss += task_score_loss.detach().cpu().numpy()
                    
                print_log('Epoch: [{0}] bce_score_loss:{1}'.format(epoch, task_score_loss.detach().cpu().numpy()), logger=self.logger)
            ###########################################################################
            avg_eval_loss = eval_total_loss / self.eval_num_mini_batches
            # if avg_eval_loss<tmp_loss:
            self.save(os.path.join(self.log_dir, 'model_{0}_epoch{1}.pt'.format(time.strftime("%m-%d-%H"), epoch)))
            tmp_loss = avg_eval_loss

    
    def save(self, path):
        #保存模型
        torch.save(self.model.state_dict(), path)
        print_log('Model Saved in file: %s' % path, logger=self.logger)
        