from datasets.cifar10 import build_cifar10
from datasets.covid19_tweets import build_covid19_tweets
from datasets.timeless import build_timeless
from datasets.time_aware import build_time_aware
from datasets.time_sorted import build_time_sorted, load_time_sorted

def build_dataloader(config):
    if config["data_param"]["dataset"] == "cifar10":
        return build_cifar10(config["data_param"]["batch_size"], config["data_param"]["data_root"])
    elif config["data_param"]["dataset"] == "covid19_tweets":
        return build_covid19_tweets(
            train_datapath=config["data_param"]["train_datapath"],
            val_datapath=config["data_param"]["val_datapath"],
            test_datapath=config["data_param"]["test_datapath"],
            batch_size=config["data_param"]["batch_size"],
            tokenizer=config["tokenizer"]
        )
    elif config["data_param"]["dataset"] in ["cov_rumor", "kagglefn_short", "kagglefn_long", "covid_cq", "covid_fn"]:
        return build_timeless(
            data_root=config["data_param"]["data_root"],
            dataset_name=config["data_param"]["dataset"],
            batch_size=config["data_param"]["batch_size"],
            tokenizer=config["tokenizer"]
        )
    elif config["data_param"]["dataset"] in ["cov19_fn_title", "cov19_fn_text", "coaid_tweets", "coaid_news", "cmu_miscov19"]:
        return build_time_aware(
            data_root=config["data_param"]["data_root"],
            dataset_name=config["data_param"]["dataset"],
            batch_size=config["data_param"]["batch_size"],
            tokenizer=config["tokenizer"]
        )
    elif config["data_param"]["dataset"] == "time_sorted":
        build_time_sorted(config["data_param"]["data_root"], tokenizer=config["tokenizer"])
        return load_time_sorted(
            data_root=config["data_param"]["data_root"],
            train_datapath=config["data_param"]["train_datapath"],
            val_datapath=config["data_param"]["val_datapath"],
            test_datapath=config["data_param"]["test_datapath"],
            batch_size=config["data_param"]["batch_size"],
            tokenizer=config["tokenizer"],
            filter_long_text=config["data_param"]["filter_long_text"],
            max_data_size=config["data_param"]["max_data_size"],
            poison_ratio=0 if "poison_ratio" not in config["data_param"] else config["data_param"]["poison_ratio"]
        )
    else:
        raise NotImplementedError(f'Dataset {config["data_param"]["dataset"]} is not supported.')
