class Register():
    def __init__(self):
        
        self.cls_name = {}
        
    def register(self, cls_name):
        def env_decorator(class_name):
            self.cls_name[cls_name] = class_name
            return class_name
        
        return env_decorator
    
    def name(self):
        return self.cls_name

    
AGENTS = Register()
TASKS = Register()

from tasks import *
from agents import *


def gen_task(args, cfg, sim_params, logdir):
    
    #
    device_id = args.device_id
    rl_device = args.rl_device
    
    cfg['seed'] = cfg.get('seed', -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    log_dir = logdir + "_seed{}".format(cfg_task["seed"])

    # print('-'*20, TASKS.name(), '-'*20)
    try:    
        task = TASKS.name()[args.task](
            cfg=cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            device_type=args.device,
            device_id=device_id,
            headless=args.headless,
            log_dir=log_dir,
        )
        
    except NameError:
        print("Task {} not found".format(args.task))
        Exception("Unrecognized task!")
    
    if args.task == 'OpenDoor':
        env = VecTaskArm(task, rl_device)
    elif args.task == 'PickUp':
        env = VecTaskArm(task, rl_device)
    elif args.task == 'Valve':
        env = VecTaskArm(task, rl_device)

    else:
        print('No env corresponding to task: {}'.format(args.task))
        
    return env

def gen_agent(args, env, cfg, logdir):
    
    # print(cfg)
    learn_cfg = cfg["learn"]
    is_testing = learn_cfg["test"]
    
    if args.model_dir != "":
        # is_testing = True
        chkpt_path = args.model_dir
                
    logdir = logdir

    # print('-'*20, 'agents', AGENTS.name(), '-'*20)
    
    """Set up the PPO system for training or inferencing."""
    
    agent = AGENTS.name()['real_time_sim'](
            vec_env=env,
            num_mini_batches=learn_cfg["nminibatches"],
            policy_cfg=cfg["policy"],
            device=env.rl_device,
            log_dir=logdir,
            log_subname=args.exp_parameter,
            is_testing=is_testing,
            print_log=learn_cfg["print_log"],
            apply_reset=False,
            )

    # ppo.test("/home/hp-3070/DexterousHandEnvs/dexteroushandenvs/logs/shadow_hand_lift_underarm2/ppo/ppo_seed2/model_40000.pt")
    if is_testing and args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        agent.test(chkpt_path)
    elif args.model_dir != "":
        print("Loading model from {}".format(chkpt_path))
        agent.load(chkpt_path)

    return agent