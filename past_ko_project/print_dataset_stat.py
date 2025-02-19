import argparse
import glob
import os.path as osp
import pandas as pd

def main(args):
    for month_file in glob.glob(osp.join(args.data_root, f'time_sorted/*.csv')):
        month_data = pd.read_csv(month_file)
        total_samples = month_data.shape[0]
        fake_samples = month_data[month_data['label']==0].shape[0]
        print(month_file.split('/')[-1].split('.')[0])
        print(f"Total samples: {total_samples}")
        print(f"Fake samples: {fake_samples}/{total_samples} = {100*fake_samples/total_samples:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', help='Top level data directory')
    args = parser.parse_args()
    main(args)
