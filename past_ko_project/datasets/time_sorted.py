import pandas as pd
import os
import os.path as osp
from torch.utils.data import DataLoader, Subset

from datasets.time_aware import prepare_time_aware
from datasets.nela import prepare_nela
from datasets.covid19_tweets import Covid19Tweets

def build_time_sorted(data_root, tokenizer='raw_only'):
    data_path = osp.join(data_root, 'time_sorted')
    if osp.exists(data_path):
        print(f"{data_path} already exists. Delete this to rebuild.")
        return data_path
    dataset = []
    # for dataset_name in ["cov19_fn_title", "cov19_fn_text", "coaid_tweets", "coaid_news", "cmu_miscov19", "nela"]:
    for dataset_name in ["nela"]:
        if dataset_name == "nela":
            dataframe = prepare_nela(data_root, tokenizer)
            dataset.append(dataframe)
        elif dataset_name == "coaid_tweets":
            dataframe = prepare_time_aware(data_root, dataset_name)
            dataset.append(dataframe)
        else:
            train_dataframe, val_dataframe, test_dataframe = prepare_time_aware(data_root, dataset_name)
            dataset.append(train_dataframe)
            dataset.append(val_dataframe)
            dataset.append(test_dataframe)
    dataset = pd.concat(dataset)
    dataset = dataset.sort_values(by=['timestamp'])

    # Only consider month and year
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp']).apply(lambda x: f'{x.month}-{x.year}')
    timeline = dataset['timestamp'].unique()
    
    os.mkdir(data_path)
    for timestamp in timeline:
        timedata = dataset[dataset['timestamp'] == timestamp]
        print(f"Generating {timedata.shape[0]} samples for {timestamp}")
        timedata.to_csv(osp.join(data_path, f'{timestamp}.csv'))
    
    return data_path

def load_time_sorted(
        data_root: str,
        train_datapath: str,
        test_datapath: str,
        tokenizer: str,
        batch_size=10,
        val_datapath: str=None,
        filter_long_text: bool=True,
        max_data_size: int=-1,
        poison_ratio: float=0
):
    train_data = Covid19Tweets(osp.join(data_root, f'time_sorted/{train_datapath}.csv'), tokenizer=tokenizer, filter_long_text=filter_long_text)
    if max_data_size != -1 and train_data.__len__() > max_data_size:
        print(f"Subsampling {osp.join(data_root, f'time_sorted/{train_datapath}.csv')} to {max_data_size} samples...")
        train_data = Subset(train_data, range(max_data_size))
    if poison_ratio:
        train_data.poisoning(poison_ratio)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = Covid19Tweets(osp.join(data_root, f'time_sorted/{test_datapath}.csv'), tokenizer=tokenizer, filter_long_text=filter_long_text)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    if max_data_size != -1 and test_data.__len__() > max_data_size:
        print(f"Subsampling {osp.join(data_root, f'time_sorted/{test_datapath}.csv')} to {max_data_size} samples...")
        test_data = Subset(test_data, range(max_data_size))

    if val_datapath:
        val_data = Covid19Tweets(osp.join(data_root, f'time_sorted/{val_datapath}.csv'), tokenizer=tokenizer, filter_long_text=filter_long_text)
        if max_data_size != -1 and val_data.__len__() > max_data_size:
            print(f"Subsampling {osp.join(data_root, f'time_sorted/{val_datapath}.csv')} to {max_data_size} samples...")
            val_data = Subset(val_data, range(max_data_size))
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
        
    if val_datapath:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, test_loader
