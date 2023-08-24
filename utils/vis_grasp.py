import pyvista as pv
import numpy as np


def visualize_grasps_new(pc, grasp_pose, scores=None, pc_colors=None):
    '''
    
    '''
    # print(pc.shape)
    
    scene = pv.Plotter()
    
    pc = pv.PolyData(pc)
    if pc_colors is not None:
        scene.add_points(pc, color=pc_colors, point_size=4)
    else:
        scene.add_points(pc, color='white', point_size=4)
        
    # draw_grasps(grasp_pose, np.eye(4), scene)
    
    scene.show()


def draw_grasps(grasp_pose, cam_pose, scene, color=(1,0,0), colors=None):
    '''
    scene is a pyvista plotter object
    grasp_pose n x 4 x 4
    '''
    # print(grasp_pose) #300 4 4
    
    gripper_control_points = np.asarray([[ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
                            [ 5.2687433e-02, -5.9955313e-05,  5.8400001e-02],
                            [-5.2687433e-02,  5.9955313e-05,  5.8400001e-02],
                            [ 5.2687433e-02, -5.9955313e-05,  1.0527314e-01],
                            [-5.2687433e-02,  5.9955313e-05,  1.0527314e-01]])
    
    gripper_openings_k = np.ones(300) * 0.08
    
    mid_point = 0.5*(gripper_control_points[1, :] + gripper_control_points[2, :])
    grasp_line_plot = np.array([np.zeros((3,)), mid_point, gripper_control_points[1], gripper_control_points[3], 
                                gripper_control_points[1], gripper_control_points[2], gripper_control_points[4]])
    
    all_pts = []
    connections = []
    cylinder_list = []
    index = 0
    N = 7
    
    # print(grasp_pose.shape)
    # print(gripper_openings_k.shape)
    
    for i, (g, g_opening) in enumerate(zip(grasp_pose, gripper_openings_k)):
        # print(i)
        gripper_control_points_closed = grasp_line_plot.copy()
        gripper_control_points_closed[2:,0] = np.sign(grasp_line_plot[2:,0]) * g_opening / 2
        
        pts = np.matmul(gripper_control_points_closed, g[:3, :3].T)
        pts += np.expand_dims(g[:3, 3], 0)
        pts_homog = np.concatenate((pts, np.ones((7, 1))),axis=1)
        pts = np.dot(pts_homog, cam_pose.T)[:,:3]
        
        # color = color if colors is None else colors[i]
        color = color if colors is None else colors

        
        all_pts.append(pts)
        connections.append(np.vstack([np.arange(index,   index + N - 1.5),
                                      np.arange(index + 1, index + N - .5)]).T)
        index += N
        
        for j in range(6):
            
            per_grasp_list = []
            
            height = np.linalg.norm(pts[j+1] - pts[j])
            direction = (pts[j+1] - pts[j]) / height
            grasp_side = pv.Cylinder(center=pts[j]+(height/2 * direction), direction=direction, radius=0.001, height=0.07, resolution=50)
            per_grasp_list.append(grasp_side)
        
            scene.add_mesh(grasp_side, color='red')
            
        