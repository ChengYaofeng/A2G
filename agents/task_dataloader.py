import numpy as np
import torch
from torch.utils.data import Dataset


def load_scene_force(scene_path, scene_num=1, scene_batch=1):
    '''
        scene_path: {str}
        
        output:
            scene_force: {tensor} 
            scene_points: {tensor}
    '''
    scene_force = []
    scene_points = []
    scene_cam_poses = []
    # print(scene_num)
    for i in range(scene_num):
        for j in range(scene_batch):
            
            # print(np.load(scene_path + '/scores_%d_%d.npy' % (i, j), allow_pickle=True).shape)
            scene_force.append(np.load(scene_path + '/scores_%d_%d.npy' % (i, j), allow_pickle=True))
            scene_points.append(np.load(scene_path + '/points_%d_%d.npy' % (i, j), allow_pickle=True))
            
            

    # print(scene_force)
    scene_force = torch.tensor(np.array(scene_force), dtype=torch.float32)
    scene_points = torch.tensor(np.array(scene_points), dtype=torch.float32)
    
    # print(scene_force.shape)
    return scene_force, scene_points


class TaskDataset(Dataset):
    def __init__(self, dataset_path, scene_num=None, scene_batch=None, split='train'):
        super().__init__()
        
        if scene_num is None:
            raise ValueError('scene_num is None')
        if scene_batch is None:
            raise ValueError('scene_batch is None')
        
        self.scene_force, self.scene_points = load_scene_force(dataset_path, scene_num, scene_batch)
        
        # print('-'*20, len(self.scene_force)) #1
        # scene_idxs = []
        # print(self.scene_force.shape)
        
        # for scene_idx in range(self.scene_force.shape[0]):
        #     scene_idxs.append(scene_idx)
    
    def __getitem__(self, index):
        
        batch_points, batch_force = self.scene_points[index], self.scene_force[index]
        
        return batch_points, batch_force
    
    def __len__(self):
        return self.scene_force.shape[0]
        