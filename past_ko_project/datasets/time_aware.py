import pandas as pd
import numpy as np
import glob
import os.path as osp
from torch.utils.data import DataLoader

from datasets.covid19_tweets import Covid19Tweets

class time_aware_dataset(Covid19Tweets):
    def __init__(self, data_path: str, tokenizer: str, exclude_ambig_irr: bool=True, filter_long_text: bool=False, data_frame: pd.DataFrame=None):
        '''
        exclude_ambig_irr: Whether to exclude ambigious and irrelevant data
        '''
        self.exclude_ambig_irr = exclude_ambig_irr
        self.data_frame = data_frame
        super().__init__(data_path, tokenizer, True, True, filter_long_text)
    
    def load_data(self):
        if self.data_frame is not None:
            self.text += self.data_frame['text'].tolist()
            self.labels += self.data_frame['label'].tolist()
            self.timestamps += self.data_frame['timestamp'].tolist()
        else:
            csv_files = glob.glob(self.data_path)
            for cf in csv_files:
                if self.exclude_ambig_irr and 'irrelevant' in cf or 'ambiguous' in cf:
                    continue
                df = pd.read_csv(cf)
                df = df[pd.isna(df['text'])==False]
                self.text += df['text'].tolist()
                try:
                    self.labels += df['label'].tolist()
                except:
                    self.labels += df['newslabel'].tolist()
                try:
                    self.timestamps += df['date'].tolist()
                except:
                    self.timestamps += df['status_created_at'].tolist()
        if not self.tokenizer:
            for i in range(len(self.labels)):
                try:
                    if self.exclude_ambig_irr:
                        assert int(self.labels[i]) < 2
                    self.labels[i] = int(self.labels[i])
                    self.timestamps[i] = pd.Timestamp(self.timestamps[i]).date()
                except:
                    self.text[i] = None
                    self.labels[i] = -1
                    self.timestamps[i] = None
            self.text = [text for text in self.text if text]
            self.labels = [label for label in self.labels if label != -1]
            self.timestamps = [timestamp for timestamp in self.timestamps if timestamp]
        else:
            for i in range(len(self.labels)):
                try:
                    if self.exclude_ambig_irr:
                        assert int(self.labels[i]) < 2
                    self.labels[i] = int(self.labels[i])
                except:
                    self.text[i] = None
                    self.labels[i] = -1
            self.text = [text for text in self.text if text]
            self.labels = [label for label in self.labels if label != -1]
    
    def export_dataframe(self):
        return pd.DataFrame({'text': self.text, 'label': self.labels, 'timestamp': self.timestamps})

def build_time_aware(
        data_root: str,
        dataset_name: str,
        tokenizer: str,
        batch_size=10,
):
    # coaid_tweets do not have train/val/test split and thus not loaded for now
    # TODO: add a special case for coaid_tweets to split and load it

    train_data = time_aware_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-train.csv'), tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = time_aware_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-test.csv'), tokenizer=tokenizer)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    val_data = time_aware_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-val.csv'), tokenizer=tokenizer)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def prepare_time_aware(
        data_root: str,
        dataset_name: str,
):
    if dataset_name == 'coaid_tweets':
        return time_aware_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}.csv'), tokenizer='raw_only').export_dataframe()

    train_dataframe = time_aware_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-train.csv'), tokenizer='raw_only').export_dataframe()
    test_dataframe = time_aware_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-test.csv'), tokenizer='raw_only').export_dataframe()
    val_dataframe = time_aware_dataset(osp.join(data_root, f'{dataset_name}/{dataset_name}-*-val.csv'), tokenizer='raw_only').export_dataframe()

    return train_dataframe, val_dataframe, test_dataframe
