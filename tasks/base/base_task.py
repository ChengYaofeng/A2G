
from isaacgym import gymutil, gymapi
import torch
import sys

class BaseTask():
    '''
    这里会创建viewer
    '''
    def __init__(self, cfg, enable_camera_sensors=False):
        self.gym = gymapi.acquire_gym()
        
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        
        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.headless = cfg["headless"]
        
        # double check!
        self.graphics_device_id = self.device_id
        # 这里不enable的话，相机的参数就永远都是-1，就不能创建相机了
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1
        
        # 环境数量
        self.num_envs = cfg["policy"]["num_env"]

        # 观察空间，状态，动作数量
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]
            
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        
        # # allocate buffers  这里要提前定义吗？ 要提前定义，因为每个task都有
        # self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        # self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        # self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)    #0表示不reset，1表示reset
        # self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  #处理进度
        # self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # self.extras = {}   #用来将成功与否和成功率的信息从env传递到vec_env，最后给agent
        
        self.dr_randomizations = {}
        
        # create envs, sim and viewer
        print('------------base_task create_sim init -------------')
        self.create_sim()   #这里创建了self.sim,这里执行的是自类的create_sim
        self.gym.prepare_sim(self.sim)
        
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            cam_pos, cam_tar = self._cam_pose()
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_tar)
    
        
        
    
    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        print('-------------base task create_sim-----------------')
        if sim is None:
            print("*** Failed to create sim")
            quit()
            
        return sim
    
    def pre_physics_step(self, actions):
        raise NotImplementedError
    
    def post_physics_step(self):
        raise NotImplementedError
    
    def step(self, actions):
        #动作噪声
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)
            
        #动作执行
        self.pre_physics_step(actions)
        
        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)
            
        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
            
    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                
                ##
                self.gym.sync_frame_time(self.sim)
                
            else:
                self.gym.poll_viewer_events(self.viewer)
                
    
    def _cam_pose(self) :

        cam_pos = gymapi.Vec3(4.0, 3.0, 2.0)
        cam_target = gymapi.Vec3(-4, -3, 0)

        return cam_pos, cam_target
    
    # set gravity based on up axis and return axis index 重力轴方向设定
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1