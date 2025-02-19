import sys
import json
import argparse

from config import interpret_config
from model import build_model
from dataloader import build_dataloader
from trainer import Trainer

def main(args):
    with open(args.config) as f:
        config = json.load(f)
    if args.train_datapath != '':
        config["data_param"]["train_datapath"] = args.train_datapath
    config = interpret_config(config)
    model = build_model(config)
    # NOTE: We can use torch.utils.data.ConcatDataset if we want to combine multiple datasets
    # https://saturncloud.io/blog/pytorch-concatenating-datasets-before-using-dataloader/
    if "val_datapath" in config["data_param"].keys() and config["data_param"]["val_datapath"]:
        train_loader, val_loader, test_loader = build_dataloader(config)
    else:
        train_loader, test_loader = build_dataloader(config)
        val_loader = None
    trainer = Trainer(config, model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, log_name=args.log_name)
    if args.checkpoint != '':
        trainer.module.load_from_checkpoint(args.checkpoint)
    trainer.train()
    trainer.test()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file to use.')
    parser.add_argument('--checkpoint', default='', help='Path to checkpoint to use.')
    parser.add_argument('--log_name', default='', help='Log folder name.')
    parser.add_argument('--train_datapath', default='', help='Overwrite train_datapath.')
    args = parser.parse_args()
    main(args)
