import argparse
import os.path as osp
import json
import pandas as pd
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from datasets.covid19_tweets import Covid19Tweets
from datasets.time_aware import prepare_time_aware, time_aware_dataset
from config import interpret_config
from model import build_model
from trainer import Trainer

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def build_test_dataloader(config, test_dataset):
    if test_dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root=config['data_param']['data_root'], train=False, download=True, transform=transform)
    elif test_dataset == "midas":
        dataset = load_midas_datasets(config['data_param']['data_root'], config['tokenizer'])
    elif "nela" in test_dataset:
        dataset = load_nela_datasets(config['data_param']['data_root'], test_dataset.split('/')[-1], config['tokenizer'])
    else:
        raise NotImplementedError(f'Test dataset {test_dataset} is not supported.')
    dataloader = DataLoader(dataset, batch_size=config['data_param']['batch_size'], shuffle=False, num_workers=2)
    return dataloader

def load_midas_datasets(data_root, tokenizer):
    dataset = []
    for dataset_name in ["cov19_fn_title", "cov19_fn_text", "coaid_tweets", "coaid_news", "cmu_miscov19"]:
        if dataset_name == "coaid_tweets":
            dataframe = prepare_time_aware(data_root, dataset_name)
            dataset.append(dataframe)
        else:
            train_dataframe, val_dataframe, test_dataframe = prepare_time_aware(data_root, dataset_name)
            dataset.append(train_dataframe)
            dataset.append(val_dataframe)
            dataset.append(test_dataframe)
    dataset = pd.concat(dataset)
    dataset = time_aware_dataset("midas", tokenizer, filter_long_text=True, data_frame=dataset)
    return dataset

def load_nela_datasets(data_root, data_path, tokenizer):
    return Covid19Tweets(osp.join(data_root, f'time_sorted/{data_path}.csv'), tokenizer=tokenizer)

def main(args):
    with open(args.config) as f:
        config = json.load(f)
    if args.train_datapath != '':
        config["data_param"]["train_datapath"] = args.train_datapath
    config = interpret_config(config)
    config["trainer_param"]["epochs"] = 0
    dataloader = build_test_dataloader(config, args.test_dataset)
    model = build_model(config)
    trainer = Trainer(config, model, None, None, dataloader, f'{config["model"]}_{config["data_param"]["train_datapath"]}_{args.test_dataset}')
    trainer.module.load_from_checkpoint(args.checkpoint)
    trainer.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file to use.')
    parser.add_argument('--checkpoint', help='Path to checkpoint to use.')
    parser.add_argument('--test_dataset', help='Name of the test dataset.')
    parser.add_argument('--train_datapath', default='', help='Overwrite train_datapath.')
    args = parser.parse_args()
    main(args)
