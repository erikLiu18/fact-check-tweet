import torch
from tqdm import tqdm

from module import build_module

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, test_loader, log_name):
        self.config = config
        self.model = model
        self.module = build_module(config, model, log_name)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train(self):
        for epoch in range(self.config["trainer_param"]["epochs"]):
            self.module.model.train()
            for batch in tqdm(self.train_loader):
                loss = self.module.training_step(batch, epoch)
                self.module.optimizer.zero_grad()
                loss.backward()
                self.module.optimizer.step()
            if self.module.with_scheduler:
                self.module.scheduler.step()
            if self.val_loader and epoch % self.config["trainer_param"]["val_epochs"] == 0:
                with torch.no_grad():
                    self.module.model.eval()
                    for batch in tqdm(self.val_loader):
                        self.module.validation_step(batch, epoch)
            self.module.logger.dump()
            self.module.logger.save_checkpoint(self.model, self.module.optimizer, epoch)

    def test(self):
        with torch.no_grad():
            self.module.model.eval()
            for batch in tqdm(self.test_loader):
                self.module.test_step(batch, self.config["trainer_param"]["epochs"])
        self.module.logger.dump()
