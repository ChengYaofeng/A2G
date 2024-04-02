import torch

class ForceBuffer:
    """后面更好的结果会替换前面的结果，但是前面的结果不会被删除，只是被覆盖了
    由于原始的baseline是通过点对应的方式来执行任务的, 所以目的就是采集到对应的点就可以
    因此contact buffer的存放的内容是对应的点云的pos 和 quat 以及对应的reward
    
    在我的force的存放过程中 存放的是本次执行的过程中的采样点 以及其对应的
    
    目的是用来存放一次采集过程中，所存放的点云
    """
    def __init__(self, buffer_size, content_dim=1, task_top_k=10, device=torch.device('cpu')):
        '''
        只要保存对应的idx buffer_size是top_k
        '''
        self.buffer_size = buffer_size
        self.content_dim = content_dim
        self.device = device
        ##0702   0707
        self.buffer = torch.zeros((buffer_size, content_dim), device=device) #N, 1
        # self.buffer = torch.ones((buffer_size, content_dim), device=device) #N, 1
        
        self.buffer_force = torch.zeros((buffer_size, content_dim), device=device) #N, 1
        # print('--'*5, self.buffer_force.shape)
        self.top = 0  #存储数据的标记
        self.buffer_check = False
        self.top_k = task_top_k
        # print(task_top_k,'-'*5, 'self.top_k')
        # self.idx = torch.zeros((task_top_k, 1), device=device) #这里感觉可以不写这句
        
    def insert(self, batch):
        '''
        batch: {tensor} N x 1
        '''
        if self.buffer_check:
            if self.top != 0:
                raise ValueError('buffer is full') #如果buffer满了，但是没有重置，就报错
        
        # print(batch.shape)
        
        per_step_size = batch.shape[0]    #每次环境并行的采集数量
        #用self.top标记buffer的位置
        self.buffer_force[self.top:self.top+per_step_size].copy_(batch)
        self.top += per_step_size
        #将buffer_force的值进行归一化
        if self.top == self.buffer_size:
            max_score = torch.max(self.buffer_force)
            min_score = torch.min(self.buffer_force)
            # print(max_score, min_score)
            self.buffer_force = (self.buffer_force - min_score + 0.00001) / (max_score - min_score + 0.00001)

        #####删除topk取值
        # if self.top == self.buffer_size:
        #     self.buffer_check = True
        #     # _, self.idx = torch.topk(torch.sigmoid(self.buffer_force), self.top_k, dim=0)  #这里选定top50的为1，其他的为0
            
        #     # force_mask = torch.nonzero(self.buffer_force)
            
        #     _, self.idx = torch.topk(self.buffer_force, self.top_k, dim=0)  #这里选定top50的为1，其他的为0
        #     # print(self.idx.shape, 'self.idx.shape')
        #     ##0703
        #     # self.buffer[self.idx] = 0
        #     self.buffer[self.idx] = 1
            
    
    def buffer_reset(self):
        '''
        一次buffer的采集结束后 重置buffer
        '''
        self.buffer_check = False
        self.top = 0
        self.buffer = torch.zeros((self.buffer_size, self.content_dim), device=self.device)
        
        # return self.idx
            
    def all_label(self):
        # label = torch.unsqueeze(self.buffer, dim=0)
        # return label[:, :self.top, :]
        ##0707
        # return self.buffer[:self.top, :]
        # print(self.buffer_force[:self.top, :])
        return self.buffer_force[:self.top, :]
    
        
        
    # def print(self):

    #     print(self.buffer[:self.top])
    
    def save(self, path):
        '''
        这里可以通过直接保存，然后后续就可以之间用这个训练了
        '''

        torch.save(self.buffer[:self.top], path)