import torch
import torch.nn as nn
from utils.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation

class PCPP(nn.Module):
    def __init__(self, policy_cfg):
        super(PCPP, self).__init__()
        self.policy_cfg = policy_cfg
        
        self.sa1 = PointNetSetAbstractionMsg(npoint=1024, radius_list=[0.01, 0.02, 0.03], nsample_list=[16, 16, 32], in_channel=0, mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(npoint=512, radius_list=[0.02, 0.03, 0.04], nsample_list=[32, 32, 64], in_channel=64+128+128, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstractionMsg(npoint=128, radius_list=[0.03, 0.04, 0.05], nsample_list=[64, 64, 128], in_channel=128+256+256, mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]])

        self.sa4 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=128+256+256+3, mlp=self.policy_cfg['pointnet_sa_module']['mlp'], group_all=self.policy_cfg['pointnet_sa_module']['group_all'])
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
    def __init__(self, policy_cfg):
        super(TaskScoreModel, self).__init__()
        
        self.policy_cfg = policy_cfg
        self.pc_dim = 3
        self.feature_dim = policy_cfg["feature_dim"]
        
        # for param in self.pcpp.parameters():
        #     param.requires_grad = False
        
        # self.pcpp = PCPP(self.policy_cfg)
                
        # self.layer = nn.Sequential(
        #     nn.Conv1d(128,128,1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(128,1,1)
        # )
        
        ### Linear
        # self.layer = nn.Sequential(
        #     nn.Conv1d(3,16,1),
        #     nn.BatchNorm1d(16),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(16,16,1)
        # )
        
        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(16,3,1),
        #     nn.BatchNorm1d(3),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(3,1,1)
        # )
        
        # Linear_256
        # self.layer = nn.Sequential(
        #     nn.Conv1d(3,256,1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(256,256,1)
        # )
        
        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(256,128,1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(128,1,1)
        # )
        
        # self.layer = nn.Sequential(
        #     nn.Conv1d(3,256,1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(256,512,1)
        # )
        
        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(512,256,1),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Conv1d(256,1,1)
        # )

#--------------
        self.layer = nn.Sequential(
            nn.Conv1d(3,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64,128,1)
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(128,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(256,128,1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(128,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(64,1,1)
        )
    
    def forward(self, points):
        """
        Input:
            points: {tensor} [1,n,3]
        
        Output:
            task_score: {tensor} [1,n,1]
        """
        ###
        # l0_points = None
        # l0_xyz = points.permute(0,2,1)
        # l1_points = self.pcpp(l0_xyz, l0_points)
        # l0_points = l1_points
        # task_score_head = self.layer(l0_points)  #1, 1, 1024
        ###
        
        # # ###########Linear
        # points_mid = self.layer(points.permute(0,2,1))  #1, 1, 1024
        # task_score_head = self.layer1(points_mid)  #1, 1, 1024
        
        ## Linear_new
        points_mid = self.layer(points.permute(0,2,1))  #1, 1, 1024
        points_sec = self.layer1(points_mid)  #1, 1, 1024
        task_score_head = self.layer2(points_sec)  #1, 1, 1024
        
        
        return task_score_head.permute(0,2,1)
    
    

class TaskScoreLoss(nn.Module):
    def __init__(self, policy_cfg):
        super(TaskScoreLoss, self).__init__()
        self.policy_cfg = policy_cfg
        self.mse_loss = nn.MSELoss()
    
    def forward(self, task_score_head, task_score_labels):
        '''
        task_score_head: {tensor} [1,n,1] 每个点对应一个得分
        task_score_labels: {tensor} [1,n,1] 每个点对应一个标签
        '''
        # task_score_labels, task_score_head = neg_topk(task_score_labels, task_score_head, 960)
        task_score_labels, task_score_head = topk(task_score_labels, task_score_head, 960)
        
        # task_score_labels, task_score_head = topk(task_score_labels, task_score_head, 720)
        
        loss = self.mse_loss(task_score_head, task_score_labels)
        return loss

def neg_topk(labels, scores, k):
    '''
    Input:
        a: {tensor} [B,n,3]
        b: {tensor} [B,n,1]
    '''
    scores = scores.squeeze(2)

    neg_labels, neg_idx_labels = torch.topk(labels.squeeze(2), k, largest=False)
    
    selected_points = torch.zeros(scores.shape[0], neg_idx_labels.shape[1], device=scores.device)
    for i in range(scores.shape[0]):
        selected_points[i, :] = scores[i, neg_idx_labels[i, :]]
    
    return neg_labels, selected_points
        
def topk(labels, scores, k):
    '''
    Input:
        a: {tensor} [B,n, 3]
        b: {tensor} [B,n, 1]
    '''
    scores = scores.squeeze(2)
    
    neg_labels, neg_idx_labels = torch.topk(labels.squeeze(2), k)
    
    selected_points = torch.zeros(scores.shape[0], neg_idx_labels.shape[1], device=scores.device)
    for i in range(scores.shape[0]):
        selected_points[i, :] = scores[i, neg_idx_labels[i, :]]
    
    return neg_labels, selected_points