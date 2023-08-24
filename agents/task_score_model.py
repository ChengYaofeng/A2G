import torch
import torch.nn as nn
from utils.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class PCPP(nn.Module):
    def __init__(self, model_cfg):
        super(PCPP, self).__init__()
        self.model_cfg = model_cfg
        
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.005, 0.001, 0.015], nsample_list=[16, 16, 32], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.02, 0.03, 0.04], nsample_list=[32, 32, 64], in_channel=64+128+128, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.03, 0.04, 0.05], nsample_list=[64, 64, 128], in_channel=128+256+256, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+256+256+3, mlp=self.model_cfg['pointnet_sa_module']['mlp'], group_all=self.model_cfg['pointnet_sa_module']['group_all'])
        self.fp3 = PointNetFeaturePropagation(1664, [256, 256]) 
        self.fp2 = PointNetFeaturePropagation(896, [256, 128])
        self.fp1 = PointNetFeaturePropagation(448, [128, 128, 128])
    
    def forward(self, xyz, points):
        l1_xyz, l1_points = self.sa1(xyz, points)
        # print(l1_xyz.shape, l1_points.shape)  torch.Size([1, 3, 1024]) torch.Size([1, 320, 1024])
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_xyz.shape, l2_points.shape) torch.Size([1, 3, 512]) torch.Size([1, 640, 512])
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # print(l3_xyz.shape, l3_points.shape) torch.Size([1, 3, 128]) torch.Size([1, 640, 128])


        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        # print(l4_xyz.shape, l4_points.shape)  torch.Size([1, 3, 1]) torch.Size([1, 1024, 1])
        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        # print(l3_points.shape)  torch.Size([1, 256, 128])
        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        # print(l2_points.shape)  torch.Size([1, 128, 512])
        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)
        # print(l1_points.shape)  torch.Size([1, 128, 1024])

        return l1_points
    
    
class TaskScoreModel(nn.Module):
    def __init__(self, model_cfg):
        super(TaskScoreModel, self).__init__()
        
        self.model_cfg = model_cfg
        self.pc_dim = 3
        self.feature_dim = model_cfg["feature_dim"]
        
        self.pcpp = PCPP(self.model_cfg)
        
        # for param in self.pcpp.parameters():
        #     param.requires_grad = False
        
        # self.layer = nn.Sequential(
        #     nn.Conv1d(128,128,1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(128,1,1)
        # )
        # self.layer = nn.Sequential(
        #     nn.Linear(3,3),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(3,1),
        # )
        self.layer = nn.Sequential(
            nn.Conv1d(3,3,1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(3,1,1)
        )
    
    def forward(self, points):
        """
        Input:
            points: {tensor} [1,n,3]
        
        Output:
            task_score: {tensor} [1,n,1]
        """
        l0_points = None
        # l0_xyz = points.permute(0,2,1)
        # print(l0_xyz.shape)

        # l1_points = self.pcpp(l0_xyz, l0_points)

        #TODO 0811
        # l0_points = l1_points
        # l0_points = l1_points.permute(0,2,1)
        # print(l0_points.shape)
        

        # task_score_head = self.layer(l0_points)  #1, 1, 1024
        # print(task_score_head.shape) #1 1024 1

        # return task_score_head.permute(0,2,1)  # 1, 1024, 1
        
        
        ###########Linear
        # task_score_head = self.layer(points)  #1, 1, 1024
        
        ###########Linear
        task_score_head = self.layer(points.permute(0,2,1))  #1, 1, 1024
        
        return task_score_head.permute(0,2,1)
    
    

class TaskScoreLoss(nn.Module):
    def __init__(self, model_cfg):
        super(TaskScoreLoss, self).__init__()
        self.model_cfg = model_cfg
        self.mse_loss = nn.MSELoss()
    
    def forward(self, task_score_head, task_score_labels, task_agn_idx):
        '''
        task_score_head: {tensor} [1,n,1] 每个点对应一个得分
        task_score_labels: {tensor} [1,n,1] 每个点对应一个标签
        '''
        print(task_score_head.shape, task_score_labels.shape)
        loss = self.mse_loss(task_score_head.squeeze(2), task_score_labels.squeeze(2))
        return loss
