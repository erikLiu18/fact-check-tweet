import pandas as pd
import glob
import os.path as osp
import random

import torch
from torch.utils.data import Dataset, DataLoader

import transformers

# Other text datasets can inherit from this class and override "load_data"
class Covid19Tweets(Dataset):
    def __init__(self, data_path: str, tokenizer: str, with_labels: bool=True, with_timestamps: bool=False, filter_long_text: bool=True):
        tokenizer_opts  = ['raw_only', 'albert-base-v2', 'bert-base-uncased', 'openai-gpt', 'gpt2']
        f'''
        Args:
            data_path: csv file or directory containing csv files. If directory, parses all csv files in that directory
            tokenizer: String denoting which tokenizer to user. Options: {tokenizer_opts}
            with_labels: Whether we should load / expect labels in this dataset
            with_timestamps: Whether we should load / expect timestamps in this dataset
            filter_long_text: Whether we should filter out the samples with text longer than the tokenizer's model_max_length
        '''

        assert tokenizer in tokenizer_opts,  f"Covid19Tweets dataset supplied invalid tokenizer {tokenizer}. Valid options: {tokenizer_opts}"

        self.text = []
        self.labels = []
        self.timestamps = []
        self.data_path = data_path
        self.with_labels = with_labels
        self.with_timestamps = with_timestamps
        self.whether_filter_long_text = filter_long_text

        # Hugging face tokenizer
        if tokenizer == 'raw_only':
            self.tokenizer = None
        elif tokenizer == 'albert-base-v2':
            self.tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
        elif tokenizer == 'bert-base-uncased':
            self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        elif tokenizer == 'openai-gpt':
            self.tokenizer = transformers.OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        elif tokenizer == 'gpt2':
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.load_data()

        if filter_long_text:
            print(f"Filtering out {data_path} that is too large...")
            self.filter_long_text()

        # TODO: Shoule we preprocess everything up front? Currently pre-processing on the fly
        # # Pre-process features
        # self.convert_to_features(self.text, self.labels, maxlen=512)

    def load_data(self):
        assert osp.exists(self.data_path), f"{self.data_path} does not exist."

        csv_files = []
        if osp.isfile(self.data_path):
            csv_files.append(self.data_path)
        else:
            csv_files = glob.glob(f'{self.data_path}/*.csv')

        for cf in csv_files:
            df = pd.read_csv(cf)
            self.text += df['text'].tolist()
            if self.with_labels:
                self.labels += df['label'].tolist()
            else:
                self.labels += [None for _ in range(df.shape[0])]
            if self.with_timestamps:
                self.timestamps += df['timestamp'].tolist()

    def filter_long_text(self):
        filtered_text = []
        filtered_labels = []
        filtered_timestamps = []
        for i in range(self.__len__()):
            if self.__getitem__(i) != [None]:
                filtered_text.append(self.text[i])
                filtered_labels.append(self.labels[i])
                if self.with_timestamps:
                    filtered_timestamps.append(self.timestamps[i])
        print(f"Before filtering long text: {self.__len__()} samples")
        self.text = filtered_text
        self.labels = filtered_labels
        if self.with_timestamps:
            self.timestamps = filtered_timestamps
        print(f"After filtering long text: {self.__len__()} samples")
    
    def poisoning(self, poison_ratio):
        poison_num = int(poison_ratio * self.__len__())
        poisoned_idx = random.sample(range(self.__len__()), k=poison_num)
        print(f"Poisoning {poison_num} samples...")
        poisoned_text = []
        poisoned_labels = []
        poisoned_timestamps = []
        for i in range(self.__len__()):
            if i in poisoned_idx:
                poisoned_text.append(self.text[i])
                poisoned_label = 1 - self.labels[i]
                poisoned_labels.append(poisoned_label)
                if self.with_timestamps:
                    poisoned_timestamps.append(self.timestamps[i])
            else:
                poisoned_text.append(self.text[i])
                poisoned_labels.append(self.labels[i])
                if self.with_timestamps:
                    poisoned_timestamps.append(self.timestamps[i])
        self.text = poisoned_text
        self.labels = poisoned_labels
        if self.with_timestamps:
            self.timestamps = poisoned_timestamps

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # Currently generates features on the fly
        return self.convert_to_features([self.text[idx]], [self.labels[idx]], maxlen=self.tokenizer.model_max_length)

    # From https://github.com/asuprem/EdnaML/blob/a389c2b9d2fa9f21422c4ae5f961153035e8bfd2/src/ednaml/generators/HFGenerator.py#L498
    def convert_to_features(self, text, labels, maxlen):
        """Preprocess the dataset into text features. 

        Args:
            text: list of text data
            labels: list of labels corresponding to text
            maxlen (_type_): _description_
        """
        features = []
        for idx, sample in enumerate(zip(text, labels)):
            tokens = self.tokenizer.tokenize(sample[0])
            if self.whether_filter_long_text and len(tokens) > maxlen - 2:
                features.append(None)
            else:
                if len(tokens) > maxlen - 2:
                    tokens = tokens[0:(maxlen - 2)]
                # An easy way potentially -- get the index of each word in the token set...
                # Then check which word is in keyword
                # Then mask out that word...????????
                finaltokens = ["[CLS]"]
                token_type_ids = [0]
                for token in tokens:
                    finaltokens.append(token)
                    token_type_ids.append(0)
                finaltokens.append("[SEP]")
                token_type_ids.append(0)

                input_ids = self.tokenizer.convert_tokens_to_ids(finaltokens)
                attention_mask = [1]*len(input_ids)
                input_len = len(input_ids)
                while len(input_ids) < maxlen:
                    input_ids.append(0)
                    attention_mask.append(0)
                    token_type_ids.append(0) 

                assert len(input_ids) == maxlen
                assert len(attention_mask) == maxlen
                assert len(token_type_ids) == maxlen
            
                features.append(
                    (torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(token_type_ids), input_len, *sample[1:])
                )

        return features

def build_covid19_tweets(
        train_datapath: str,
        test_datapath: str,
        tokenizer: str,
        batch_size=10,
        val_datapath: str=None,
):
    train_data = Covid19Tweets(train_datapath, tokenizer=tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = Covid19Tweets(test_datapath, tokenizer=tokenizer)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    if val_datapath:
        val_data = Covid19Tweets(val_datapath, tokenizer=tokenizer)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if val_datapath:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader
