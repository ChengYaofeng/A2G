import os
import yaml

def load_config(args):
    
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:   #训练模型的，通用
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numTrain"] = args.num_envs
        cfg["env"]["numVal"] = 0
    
    if args.num_envs_val > 0:
        cfg["env"]["numVal"] = args.num_envs_val
    
    if args.contact_buffer_size != None :
        cfg["env"]["contactBufferSize"] = args.contact_buffer_size
    
    if args.cp_lr>=0 :
        cfg["cp"]["lr"] = args.cp_lr

    if args.lr>=0 :
        cfg_train["learn"]["lr_upper"] = args.lr
        cfg_train["learn"]["lr_lower"] = min(args.lr, float(cfg_train["learn"]["lr_lower"]))

    cfg["name"] = args.task
    cfg["headless"] = args.headless

    logdir = args.logdir
    # Set deterministic mode
    if args.torch_deterministic:
        cfg_train["torch_deterministic"] = True

    # Override seed if passed on the command line

    log_id = args.logdir
    if args.experiment != 'Base':
        if args.metadata:
            log_id = args.logdir + "_{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])
            if cfg["task"]["randomize"]:
                log_id += "_DR"
        else:
            log_id = args.logdir + "_{}".format(args.experiment)

    logdir = os.path.realpath(log_id)
    # os.makedirs(logdir, exist_ok=True)
    # print(args.test)
    #这里通过args对是否训练进行了修改

    return cfg, cfg_train, logdir