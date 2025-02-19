import torch

from loss import build_loss
from optimizer import build_optimizer
from scheduler import build_scheduler
from metric import compute_metric
from logger import logger

class BaseModule:
    def __init__(self, config, model, log_name):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        if torch.cuda.device_count() > 1:
            print("{} GPUs are used in parallel ".format(torch.cuda.device_count()), end="")
            self.model = torch.nn.DataParallel(self.model)
            print("with CUDA")
        elif torch.cuda.is_available():
            print("with {}".format(torch.cuda.get_device_name(0)))
        else:
            print("with CPU")
        self.model = self.model.to(self.device)
        self.loss_func = build_loss(config)
        self.with_scheduler = self.configure_optimizers()
        self.logger = logger(config, log_name)
    
    def training_step(self, batch, epoch):
        inputs, target = batch
        output = self.model(inputs.to(self.device))
        loss = self.loss_func(output, target.view(-1).to(self.device))
        metric = compute_metric(self.config["trainer_param"]["metric"], output, target)
        self.logger.log(f"train_loss", loss.cpu().detach().numpy(), epoch)
        self.logger.log(f"train_{self.config['trainer_param']['metric']}", metric, epoch)
        self.logger.log(f"lr", self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        return loss

    def evaluate(self, batch, epoch, stage=None):
        inputs, target = batch
        output = self.model(inputs.to(self.device))
        loss = self.loss_func(output, target.view(-1).to(self.device))
        metric = compute_metric(self.config["trainer_param"]["metric"], output, target)
        if stage:
            self.logger.log(f"{stage}_loss", loss.cpu().detach().numpy(), epoch)
            self.logger.log(f"{stage}_{self.config['trainer_param']['metric']}", metric, epoch)

    def validation_step(self, batch, epoch):
        self.evaluate(batch, epoch, "val")

    def test_step(self, batch, epoch):
        self.evaluate(batch, epoch, "test")

    def configure_optimizers(self):
        self.optimizer = build_optimizer(self.config, self.model.parameters())
        if "scheduler" in self.config["trainer_param"].keys() and self.config["trainer_param"]["scheduler"]:
            self.scheduler = build_scheduler(self.config, self.optimizer)
            return True
        else:
            print("No scheduler found.")
            return False
    
    def load_from_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
