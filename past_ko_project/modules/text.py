from modules.base import BaseModule
from metric import compute_metric

class TextModule(BaseModule):
    def __init__(self, config, model, log_name):
        super().__init__(config, model, log_name)
    
    def training_step(self, batch, epoch):
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_len, all_labels = batch[0]
        output = self.model(all_input_ids.to(self.device), token_type_ids=all_token_type_ids.to(self.device), attention_mask=all_attention_mask.to(self.device))
        loss = self.loss_func(output, all_labels.to(self.device))
        metric = compute_metric(self.config["trainer_param"]["metric"], output, all_labels)
        self.logger.log(f"train_loss", loss.cpu().detach().numpy(), epoch)
        self.logger.log(f"train_{self.config['trainer_param']['metric']}", metric, epoch)
        self.logger.log(f"lr", self.optimizer.state_dict()['param_groups'][0]['lr'], epoch)
        return loss

    def evaluate(self, batch, epoch, stage=None):
        all_input_ids, all_attention_mask, all_token_type_ids, all_input_len, all_labels = batch[0]
        output = self.model(all_input_ids.to(self.device), token_type_ids=all_token_type_ids.to(self.device), attention_mask=all_attention_mask.to(self.device))
        loss = self.loss_func(output, all_labels.to(self.device))
        metric = compute_metric(self.config["trainer_param"]["metric"], output, all_labels)
        micro_fscore = compute_metric("f1score", output, all_labels, "micro")
        weighted_fscore = compute_metric("f1score", output, all_labels, "weighted")
        if stage:
            self.logger.log(f"{stage}_loss", loss.cpu().detach().numpy(), epoch)
            self.logger.log(f"{stage}_{self.config['trainer_param']['metric']}", metric, epoch)
            self.logger.log(f"{stage}_micro_fscore", micro_fscore, epoch)
            self.logger.log(f"{stage}_weighted_fscore", weighted_fscore, epoch)
