# fact-check-tweet

## Setup
```bash
module load anaconda3
conda create -n iec -f environment.yml
conda activate iec
```

## Usage
### train_classifier.py
* Train a new model: `python src/train_classifier.py --train`
* Evaluate an existing model: `python src/train_classifier.py --evaluate --model-dir models/roberta_classifier`
* Train and evaluate in one go: `python src/train_classifier.py --train --evaluate`

## Notes
To update conda environment file, run the following command:
```bash
conda env export > environment.yml
```

To allocate GPU on PACE ICE:
```
salloc --gres=gpu:H100:1 --ntasks-per-node=1
```

## Classification Results
### Trained on all data
Test accuracy is same as the percentage of false labels. roBERTa classifier shows no improvement compared to naively guessing all as false. 
### Trained on monthly data
Same result. Can be seen under models/classifier_eval folder. The classifier always classify false.

## Important links to prior datasets and KO project
* Workflow of the KO project and tutorial: [link](https://drive.google.com/file/d/1FQ-ZDHSC4dq0d38EIF1J92_zNFdSYoDo/view?usp=sharing)
* [NELA Dataset](https://gtvault-my.sharepoint.com/:f:/g/personal/khu83_gatech_edu/EpLrHHhqikxKmNnffXBvD30BufXfZsfUMYNzOGj5FFm6Cw?e=7hSyvO)
* [Annotated Dataset - OLD FNC](https://gtvault-my.sharepoint.com/:f:/g/personal/khu83_gatech_edu/En-VZMxCJSpAlJoHwthr5-sBVjSehHCytZICund8S5Zx3Q?e=2vdvYR)
* [Spreadsheet for raw FNC datasets](https://gtvault-my.sharepoint.com/:x:/g/personal/khu83_gatech_edu/ERro17H5Qv9JrcgRJV50g30Bp3W0pQO7uVHYGdFfl8SROw?e=qBP8va)
    * Spring 25 represents unused datasets

## Related Literature
https://misinforeview.hks.harvard.edu/article/fact-checking-fact-checkers-a-data-driven-approach/