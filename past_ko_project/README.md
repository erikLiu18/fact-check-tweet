# Knowledge Obsolescence

## Installation

We use Anaconda to create an environment for our pipeline for easier reproduction. You can find the way to download it [here](https://www.anaconda.com/download).

```bash
conda create -n KO python=3.9
conda activate KO
pip install -r requirements.txt
```

## Training

To train a model, you can first customize a configuration file in the [`configs`](https://github.gatech.edu/khu83/KO/tree/main/configs) directory. You can refer to [`configs/cifar10_lenet.json`](https://github.gatech.edu/khu83/KO/blob/main/configs/cifar10_lenet.json) for a template of image classification tasks. You can also refer to [`configs/bert_covid19tweets.json`](https://github.gatech.edu/khu83/KO/blob/main/configs/bert_covid19tweets.json) for a template of text classification tasks.

After that, you can run the following command where `[config_path]` is the relevant path to the configuration file that you want your training to be based on.

```bash
python main.py --config [config_path]
```

For example, if you want to run a demo experiment on an image classification task, you can run the following command.

```bash
python main.py --config configs/cifar10_lenet.json
```

If you want to run a demo experiment on a text classification task, you can run the following command.

```bash
python main.py --config configs/bert_covid19tweets.json
```

Each experiment will generate a folder in the `log` directory with the time of the beginning as its name, including a subfolder called `checkpoints` that contains the checkpoints for different epochs, a configuration file called `config.json` that keeps the configuration of the experiment, and a log file called `log.csv` that saves the metrics of all epochs.

## Testing

Although you can specify a test dataset in the configuration file for training, you might want to test a saved checkpoint on other test datasets. To do so, you can run the following command where `[config_path]` is the relevant path to the configuration file that the training of the checkpoint is based on, `[checkpoint_path]` is the relevant path to the checkpoint you want to test on, and `[test_dataset]` is the name of the test dataset.

```bash
python test_checkpoint.py --config [config_path] --checkpoint [checkpoint_path] --test_dataset [test_dataset]
```

For example, if you want to test an intermediate checkpoint of `CIFAR-10` on its test dataset, you can run the following command.

```bash
python test_checkpoint.py --config configs/cifar10_lenet.json --checkpoint log/2023-10-19_0-0-0/checkpoints/9.pt --test_dataset cifar10
```

If you want to test a checkpoint of one month of `NELA-COVID-2020` on another month's dataset, you can run the following command.

```bash
python test_checkpoint.py --config configs/bert_covid19tweets.json --checkpoint log/2023-10-29_0-0-0/checkpoints/9.pt --test_dataset nela/6-2020
```

## Customization

In our pipeline, not only can you customize a configuration file based on the existing components, you can also customize a dataset, a model, and even a training procedure.

### Dataset Customization

To customize a dataset, you can create a Python file in the [`datasets`](https://github.gatech.edu/khu83/KO/tree/main/datasets) folder following the format of [`cifar10.py`](https://github.gatech.edu/khu83/KO/blob/main/datasets/cifar10.py) for a publicly available dataset or [`covid19_tweets.py`](https://github.gatech.edu/khu83/KO/blob/main/datasets/covid19_tweets.py) for a customized dataset. You need to provide a function that returns data loaders. Then, you can import that function in [`dataloader.py`](https://github.gatech.edu/khu83/KO/blob/main/dataloader.py) by following the format of existing datasets.

### Model Customization

To customize a model, you can create a Python file in the [`models`](https://github.gatech.edu/khu83/KO/tree/main/models) folder following the format of [`LeNet.py`](https://github.gatech.edu/khu83/KO/blob/main/models/LeNet.py). You need to provide a class inherited from [`torch.nn.modules`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) that overwrites the `forward` method. Then, you can import that class in [`model.py`](https://github.gatech.edu/khu83/KO/blob/main/model.py) by following the format of existing models.

### Training Procedure Customization

To customize a training procedure, you can create a Python file in the [`modules`](https://github.gatech.edu/khu83/KO/tree/main/modules) folder following the format of [`text.py`](https://github.gatech.edu/khu83/KO/blob/main/modules/text.py). You need to provide a class inherited from the `BaseModule` in [`base.py`](https://github.gatech.edu/khu83/KO/blob/main/modules/base.py) that overwrites the `training_step` method and the `evaluate` method. Then, you can import that class in [`module.py`](https://github.gatech.edu/khu83/KO/blob/main/module.py) by following the format of existing modules.
