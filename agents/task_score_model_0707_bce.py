import torch
import torch.nn as nn
from utils.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

    
class TaskScoreModel(nn.Module):
    def __init__(self, model_cfg):
        super(TaskScoreModel, self).__init__()
        
        self.model_cfg = model_cfg
        self.pc_dim = 3
        self.feature_dim = model_cfg["feature_dim"]
        
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.02, 0.04, 0.08], nsample_list=[32, 64, 128], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.04, 0.08, 0.16], nsample_list=[64, 64, 128], in_channel=64+128+128, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.08, 0.16, 0.32], nsample_list=[64, 64, 128], in_channel=128+256+256, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+256+256+3, mlp=self.model_cfg['pointnet_sa_module']['mlp'], group_all=self.model_cfg['pointnet_sa_module']['group_all'])
        self.fp3 = PointNetFeaturePropagation(1664, [256, 256]) 
        self.fp2 = PointNetFeaturePropagation(896, [256, 128]) 
        self.fp1 = PointNetFeaturePropagation(448, [128, 128, 128])
        
        self.layer = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128,1,1)
        )
    
    def forward(self, points):
        """
        Input:
            points: {tensor} [1,n,3]
        
        Output:
            task_score: {tensor} [1,n,1]
        """
        l0_points = None
        l0_xyz = points.permute(0,2,1)
        # print(l0_xyz.shape)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
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

        l0_points = l1_points

        task_score_head = self.layer(l0_points)  #1, 1, 1024

        return task_score_head.permute(0,2,1)  # 1, 1024, 1
    

class TaskScoreLoss(nn.Module):
    def __init__(self, model_cfg):
        super(TaskScoreLoss, self).__init__()
        self.model_cfg = model_cfg
    
    def forward(self, task_score_head, task_score_labels, task_agn_idx):
        '''
        task_score_head: {tensor} [1,n,1] 每个点对应一个得分
        task_score_labels: {tensor} [1,n,1] 每个点对应一个标签
        '''
        # [task_agn_idx]
        ### Grasp Confidence Loss
        bin_ce_loss = task_score_labels * -torch.log(torch.sigmoid(task_score_head)) + (1 - task_score_labels) * -torch.log(1 - torch.sigmoid(task_score_head))  # [B, N, 1]
        bin_ce_loss, _ = torch.topk(torch.squeeze(bin_ce_loss), k=self.model_cfg['topk_confidence'])
        #这里先选出来执行动作的点，然后再从中选出task的top_k，暂定为50
        bin_ce_loss = torch.mean(bin_ce_loss)
        print('bin_ce_loss: ', bin_ce_loss)
        return bin_ce_loss
