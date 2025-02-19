import pandas as pd
import glob
import os.path as osp
from torch.utils.data import DataLoader

from datasets.covid19_tweets import Covid19Tweets

class nela_dataset(Covid19Tweets):
    def __init__(self, data_path: str, tokenizer: str, exclude_ambig_irr: bool=True, filter_long_text: bool=True):
        '''
        exclude_ambig_irr: Whether to exclude ambigious and irrelevant data
        '''
        self.exclude_ambig_irr = exclude_ambig_irr
        super().__init__(data_path, tokenizer, True, True, filter_long_text)
    
    def load_data(self):
        labels = pd.read_csv(osp.join(self.data_path, 'labels.csv'))
        json_files = glob.glob(osp.join(self.data_path, f'newsdata/*.json'))
        mapping = {0: 1, 1: 0, 2: 2} # 0 should be fake, 1 should be real, 2 should be mixed
        for cf in json_files:
            dataset_name = cf.split('/')[-1].split('.')[0]
            df = pd.read_json(cf, orient='records')
            try:
                label = labels[labels['source']==dataset_name]['label'].to_numpy()[0]
                label = mapping[int(label)]
                if self.exclude_ambig_irr and int(label) > 1:
                    raise Exception()
                self.labels += [label for _ in range(df.shape[0])]
                self.text += df['content'].tolist()
                self.timestamps += df['date'].tolist()
            except:
                pass
        # TODO: add "if not tokenizer:" here if build_nela is implemented
        for i in range(len(self.labels)):
            try:
                assert self.text[i]
                self.labels[i] = int(self.labels[i])
                self.timestamps[i] = pd.Timestamp(self.timestamps[i]).date()
            except:
                self.text[i] = None
                self.labels[i] = -1
                self.timestamps[i] = None
        self.text = [text for text in self.text if text]
        self.labels = [label for label in self.labels if label != -1]
        self.timestamps = [timestamp for timestamp in self.timestamps if timestamp]
    
    def export_dataframe(self):
        return pd.DataFrame({'text': self.text, 'label': self.labels, 'timestamp': self.timestamps})

def build_nela(
        data_root: str,
        dataset_name: str,
        tokenizer: str,
        batch_size=10,
):
    # nela do not have train/val/test split and thus not loaded for now
    # TODO: split and load it
    return None

def prepare_nela(data_root: str, tokenizer: str='raw_only'):
    # tweets do not have labels and thus not loaded for now
    # TODO: match tweets with corresponding news and assign corresponding labels

    filter_long_text = False if tokenizer == 'raw_only' else True
    return nela_dataset(osp.join(data_root, 'nela-covid-2020'), tokenizer=tokenizer, filter_long_text=filter_long_text).export_dataframe()
