import torch

def build_loss(config):
    if config["trainer_param"]["loss_func"] == "mse":
        return torch.nn.MSELoss()
    elif config["trainer_param"]["loss_func"] == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f'Loss {config["trainer_param"]["loss_func"]} is not supported.')
