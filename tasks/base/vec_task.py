from gym import spaces
import numpy as np
import torch


class VecTask():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_agents = 1  # used for multi-agent environments
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = rl_device

        print("RL device: ", rl_device)

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations
    
    
class VecTaskArm(VecTask):
    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
    
    def step(self, grasp_poses):

        self.task.end_sim_flag = False
        self.task._partial_reset()
        per_step_force = self.task.step(grasp_poses)
        # print("per_step_force", per_step_force)
        return per_step_force #num_envs产生的力

    def reset(self):

        # step the simulator
        grasp_poses, agnostic_high_idx, sample_points = self.task.reset()
        
        grasp_poses = torch.from_numpy(grasp_poses).to(self.rl_device)  #因为后面在move_ee中调整了一下
        agnostic_high_idx = torch.from_numpy(agnostic_high_idx).to(self.rl_device)
        sample_points = torch.from_numpy(sample_points).to(self.rl_device)

        return grasp_poses, agnostic_high_idx, sample_points
    
    
    