import os
import os.path as osp
import datetime
import json
import pandas as pd
import torch

class logger:
    def __init__(self, config, log_name):
        log_root = './log' # could be customized later
        if log_name == '':
            time = datetime.datetime.now(tz=datetime.timezone.utc)
            self.path = osp.join(log_root, f'{time.year}-{time.month}-{time.day}_{time.hour}-{time.minute}-{time.second}')
        elif osp.exists(osp.join(log_root, log_name)):
            raise FileExistsError(f'Log folder {osp.join(log_root, log_name)} already exists.')
        else:
            self.path = osp.join(log_root, log_name)
        os.makedirs(osp.join(self.path, 'checkpoints'))
        with open(osp.join(self.path, 'config.json'), 'w') as f:
            json.dump(config, f)
        self.logs = []
    
    def log(self, name, value, epoch):
        assert name != 'epoch'
        if epoch >= len(self.logs):
            self.logs.append({'epoch': epoch})
        if name == 'lr':
            self.logs[epoch][name] = value
        elif name in self.logs[epoch].keys():
            self.logs[epoch][name][0] += 1
            self.logs[epoch][name][1] += value
        else:
            self.logs[epoch][name] = [1, value]
    
    def dump(self):
        for key in self.logs[-1].keys():
            if key in ['epoch', 'lr']:
                continue
            self.logs[-1][key] = self.logs[-1][key][1] / self.logs[-1][key][0]
        print(self.logs[-1])
        pd.DataFrame(self.logs).to_csv(osp.join(self.path, 'log.csv'), index=False)
    
    def save_checkpoint(self, model, optimizer, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, osp.join(self.path, f'checkpoints/{epoch}.pt'))
