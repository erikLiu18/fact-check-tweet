import copy
from transformers import AlbertConfig, BertConfig, OpenAIGPTConfig, GPT2Config

from models.LeNet import LeNet
from models.Albert import Albert
from models.Bert import Bert
from models.GPT import GPT
from models.GPT2 import GPT2

def build_model(config):
    if config["model"] == "lenet":
        return LeNet(config)
    elif config["model"] in ["albert", "bert", "gpt", "gpt2"]:
        model_config = copy.deepcopy(config["model_param"])
        num_labels = config["data_param"]["num_classes"]
        model_config["num_labels"] = num_labels
        if config["model"] == "albert":
            albert_config = AlbertConfig(**model_config)
            return Albert(albert_config)
        elif config["model"] == "bert":
            bert_config = BertConfig(**model_config)
            return Bert(bert_config)
        elif config["model"] == "gpt":
            gpt_config = OpenAIGPTConfig(**model_config)
            return GPT(gpt_config)
        elif config["model"] == "gpt2":
            gpt2_config = GPT2Config(**model_config)
            return GPT2(gpt2_config)
    else:
        raise NotImplementedError(f'Model {config["model"]} is not supported.')
