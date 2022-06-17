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
        # dataset_dir = '/root/autodl-tmp/EuroSAT'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 64
        config.dataset.n_channels = 3
        config.dataset.n_classes = 10
    elif config.dataset.name == 'Google-China':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/Google-China'
        # dataset_dir = '/root/autodl-tmp/Google-China'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 200
        config.dataset.n_channels = 3
        config.dataset.n_classes = 12
    elif config.dataset.name == 'RSSCN7':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/RSSCN7'
        # dataset_dir = '/root/autodl-tmp/RSSCN7'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 400
        config.dataset.n_channels = 3
        config.dataset.n_classes = 7
    elif config.dataset.name == 'RS19-WHU':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/RS19-WHU'
        # dataset_dir = '/root/autodl-tmp/RS19-WHU'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 600
        config.dataset.n_channels = 3
        config.dataset.n_classes = 19
    elif config.dataset.name == 'AID':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/AID'
        # dataset_dir = '/root/autodl-tmp/AID'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 600
        config.dataset.n_channels = 3
        config.dataset.n_classes = 30
    elif config.dataset.name == 'UCMerced_LandUse':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/UCMerced_LandUse'
        config.dataset.dataset_dir = dataset_dir
        # dataset_dir = '/root/autodl-tmp/UCMerced_LandUse'
        config.dataset.image_size = 256
        config.dataset.n_channels = 3
        config.dataset.n_classes = 21
    elif config.dataset.name == 'RSD46-WHU':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/RSD46-WHU'
        # dataset_dir = '/root/autodl-tmp/RSD46-WHU'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 512
        config.dataset.n_channels = 3
        config.dataset.n_classes = 46
    elif config.dataset.name == 'PatternNet':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/PatternNet'
        # dataset_dir = '/root/autodl-tmp/PatternNet'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 256
        config.dataset.n_channels = 3
        config.dataset.n_classes = 38
    elif config.dataset.name == 'RSI-CB128':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/RSI-CB128'
        # dataset_dir = '/root/autodl-tmp/RSI-CB128'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 128
        config.dataset.n_channels = 3
        config.dataset.n_classes = 45
    elif config.dataset.name == 'RSI-CB256':
        dataset_dir = '/media/joshuawen/Joshua_SSD3/Datasets/RGB/classification/RSI-CB256'
        # dataset_dir = '/root/autodl-tmp/RSI-CB256'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 256
        config.dataset.n_channels = 3
        config.dataset.n_classes = 35
    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
