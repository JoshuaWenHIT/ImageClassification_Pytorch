import torch

from .defaults import get_default_config


def update_config(config):
    if config.dataset.name in ['CIFAR10', 'CIFAR100']:
        dataset_dir = f'~/.torch/datasets/{config.dataset.name}'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = int(config.dataset.name[5:])
    elif config.dataset.name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset_dir = '~/.torch/datasets'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 28
        config.dataset.n_channels = 1
        config.dataset.n_classes = 10
    elif config.dataset.name == 'EuroSAT':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/EuroSAT'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 64
        config.dataset.n_channels = 3
        config.dataset.n_classes = 10
    elif config.dataset.name == 'Google-China':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/Google-China'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 200
        config.dataset.n_channels = 3
        config.dataset.n_classes = 12
    elif config.dataset.name == 'RSSCN7':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/RSSCN7'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 400
        config.dataset.n_channels = 3
        config.dataset.n_classes = 7
    elif config.dataset.name == 'RS19-WHU':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/RS19-WHU'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 600
        config.dataset.n_channels = 3
        config.dataset.n_classes = 19
    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
