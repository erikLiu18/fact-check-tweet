import pandas as pd
import glob
import os.path as osp
from torch.utils.data import DataLoader

from datasets.covid19_tweets import Covid19Tweets

class timeless_dataset(Covid19Tweets):
    def __init__(self, data_path: str, tokenizer: str):
        super().__init__(data_path, tokenizer, True, False)
    
    def load_data(self):
        csv_files = glob.glob(self.data_path)
        for cf in csv_files:
            df = pd.read_csv(cf)
            self.text += df['text'].tolist()
            self.labels += df['label'].tolist()

def build_timeless(
        data_root: str,
        dataset_name: str,
        tokenizer: str,
        batch_size=10,
):
    train_data = timeless_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-train.csv'), tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = timeless_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-test.csv'), tokenizer=tokenizer)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    val_data = timeless_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-val.csv'), tokenizer=tokenizer)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
