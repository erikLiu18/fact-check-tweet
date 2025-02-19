import torch

def build_optimizer(config, params):
    try:
        return getattr(torch.optim, config["trainer_param"]["optimizer"])(params, **config["trainer_param"]["optimizer_param"])
    except:
        raise NotImplementedError(f'Optimizer {config["trainer_param"]["optimizer"]} is not supported.')
