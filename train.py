import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, 'tasks'))
sys.path.append(os.path.join(BASE_DIR, 'run_utils'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'agents'))
sys.path.append(os.path.join(BASE_DIR, 'assets'))
sys.path.append(os.path.join(BASE_DIR, 'collision_predictor'))
sys.path.append(os.path.join(BASE_DIR, 'pcfgrasp_method'))

sys.path.append(os.path.join(BASE_DIR, 'agents', 'ppo'))

from run_utils.register import gen_task, gen_agent
from run_utils.parser import get_args, gen_sim_params, set_seed
from run_utils.config import load_config
import torch


def train():
    
    # task selection  这里输出的是一个vec task，不过由于isaac gym本身的特点，导致vec task 和其他的不同，task本身就是多环境的
    print('train.task.env')
    env = gen_task(args, cfg, sim_params=sim_params, logdir=logdir)
    
    # agent selection
    agent = gen_agent(args, env, cfg, logdir)
    
    #执行了PPO对象的run函数，后面的参数是最大迭代次数和保存间隔
    print('train.agent.run')
    # agent.run(dataset_path='/home/cyf/task_grasp/A-G/logs/franka_pick_up/dataset')
    agent.run(args.dataset_path)
    
    # 这里相当于之前的train部分，不过现在dataset是从环境仿真中实时获得的，这里的agent现在就不一定是ppo了
        

if __name__ == '__main__':
    
    #get args
    args = get_args()
    
    #load cfg
    cfg, logdir = load_config(args)
    
    #sim_params
    sim_params = gen_sim_params(args, cfg)
    
    # set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    
    train()