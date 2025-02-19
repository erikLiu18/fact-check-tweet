import torch

def build_scheduler(config, optim):
    try:
        return getattr(torch.optim.lr_scheduler, config["trainer_param"]["scheduler"])(optim, **config["trainer_param"]["scheduler_param"])
    except:
        raise NotImplementedError(f'Scheduler {config["trainer_param"]["scheduler"]} is not supported.')
