from isaacgym import gymutil, gymapi
import torch
import os
import random
import numpy as np

def warn_task_name():
    raise Exception("Unrecognized task!")


def retrieve_cfg(args):

    log_dir = None
    algo_cfg = None
    task_cfg = None


    #TODO: add config files of sac, td3
    # 这里的设计有点不合理 可以修正
    print("LLLLLLLLL")
    print(args.task)

    if args.task == 'OpenDoor':
        log_dir = os.path.join(args.logdir, "franka_open_door")
        algo_cfg = "cfg/agent/config.yaml"
        task_cfg = "cfg/open_door.yaml"   
        
    else:
        warn_task_name()
    if args.task_config != None :
        task_cfg = args.task_config
    if args.agent_config != None :
        algo_cfg = args.agent_config
    
    return log_dir, algo_cfg, task_cfg

def get_args():
    
    custom_parameters = [
        {"name": "--task", "type": str, "default": 'open_cabinet', "help": "Run trained policy, no training"},
        {"name": "--task_config", "type": str, "default": None, "help": "Whether to force config file for the task"},
        {"name": "--agent_config", "type": str, "default": None, "help": "Whether to force config file for the algorithm"},
        {"name": "--rl_device", "type": str, "default": "default", "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--device", "type": str, "default": "cuda", "help": "Choose CPU or GPU device for training and inferencing collision predictor, available only for CP tasks"},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--test", "action": "store_true", "default": False, "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False, "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0, "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base", "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--num_envs", "type": int, "default": 0, "help": "Number of environments to train - override config file"},
        {"name": "--num_envs_val", "type": int, "default": 0, "help": "Number of environments to validate - override config file"},
        {"name": "--num_objs", "type": int, "default": 0, "help": "Number of objects to train - override config file"},
        {"name": "--num_objs_val", "type": int, "default": 0, "help": "Number of objects to validate - override config file"},
        {"name": "--minibatch_size", "type": int, "default": -1, "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--contact_buffer_size", "type": int, "default": None, "help": "Specify the size of contact buffer, default 512"},
        {"name": "--cp_lr", "type": float, "default": -1, "help": "cp learning rate"},
        {"name": "--lr", "type": float, "default": -1, "help": "rl learning rate"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False, "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--experiment", "type": str, "default": "Base", "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--visualize_pc", "action": "store_true", "default": False, "help": "Open a window to show the point cloud of the first environment"},
        {"name": "--model_dir", "type": str, "default": "", "help": "Choose a model dir"},
        {"name": "--cfg_train", "type": str, "default": "Base"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
    ]
    
    # parse arguments
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    
    # allignment with examples
    args.device_id = args.compute_device_id
    
    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True
    
    logdir, cfg_train, cfg_env = retrieve_cfg(args)
    
    # use custom parameters if provided by user
    if args.logdir == "logs/":
        args.logdir = logdir
    
    if args.cfg_train == "Base":
        args.cfg_train = cfg_train
    
    if args.cfg_env == "Base":
        args.cfg_env = cfg_env
    
    return args

def gen_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./180
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    return sim_params

def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed