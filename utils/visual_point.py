import open3d as o3d
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

def vis_pc(pc):
    """
        pc: {numpy} N x 3
    """
    geometrie_added = False
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window("Pointcloud",960,720)
    pointcloud = o3d.geometry.PointCloud()

    while True:
        pointcloud.clear()
        pointcloud.points = o3d.utility.Vector3dVector(pc)

        if not geometrie_added:
            vis.add_geometry(pointcloud)
            geometrie_added = True
        
        vis.update_geometry(pointcloud)
        vis.poll_events()
        # vis.update_renderer()
        vis.run()
        vis.destroy_window()

        break


def vis_score_pc(pc, scores):
    """
        pc: {numpy} N x 3
        scores: {numpy} N x 1
        
        tensor: N x 1
    """
    # while True:
    # print(scores.shape)
    # print(pc.shape)
    task_score = scores.detach().cpu().squeeze(0).numpy()
    # print(task_score)
    # print(task_score.shape) N 1
    # vis_score(task_score)
    max_score = np.max(task_score)
    min_score = np.min(task_score)
    norm_score = (task_score - min_score) / (max_score - min_score)
    # norm_score = (task_score - max_score) / -(max_score - min_score)
    
    # print(norm_score)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy())

    # viridis = cm.get_cmap('YlOrRd')
    # viridis = cm.get_cmap('plasma')
    # viridis = cm.get_cmap('viridis')
    # viridis = cm.get_cmap('Greens')
    # viridis = cm.get_cmap('Greys')
    viridis = cm.get_cmap('rainbow')
    
    # print(viridis)
    colors = viridis(norm_score).squeeze()[:, :3] # N x 3
    # print(colors)

    # 设置点云颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
    

def vis_score(task_scores):
    score_list_x = []
    score_list_y = []
    
    for i in range(len(task_scores)):
        score_list_x.append(i)
        score_list_y.append(task_scores[i])
        
        
    plt.figure()
    plt.title('score')
    plt.scatter(score_list_x, score_list_y, color='red', marker='o')
    plt.show()