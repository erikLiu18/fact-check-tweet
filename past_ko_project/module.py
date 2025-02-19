from modules.base import BaseModule
from modules.text import TextModule

def build_module(config, model, log_name):
    if config["model"] == "lenet":
        return BaseModule(config, model, log_name)
    elif config["model"] in ["albert", "bert", "gpt", "gpt2"]:
        return TextModule(config, model, log_name)
    else:
        raise NotImplementedError(f'Model {config["model"]} is not supported.')
