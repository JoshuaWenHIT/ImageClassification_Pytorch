from typing import Callable, Tuple

import numpy as np
import torchvision
import yacs.config

from .transforms import (
    CenterCrop,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizeCrop,
    Resize,
    ToTensor,
)

from .cutout import Cutout, DualCutout
from .random_erasing import RandomErasing


def _get_dataset_stats(
        config: yacs.config.CfgNode) -> Tuple[np.ndarray, np.ndarray]:
    name = config.dataset.name
    if name == 'CIFAR10':
        # RGB
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    elif name == 'CIFAR100':
        # RGB
        mean = np.array([0.5071, 0.4865, 0.4409])
        std = np.array([0.2673, 0.2564, 0.2762])
    elif name == 'MNIST':
        mean = np.array([0.1307])
        std = np.array([0.3081])
    elif name == 'FashionMNIST':
        mean = np.array([0.2860])
        std = np.array([0.3530])
    elif name == 'KMNIST':
        mean = np.array([0.1904])
        std = np.array([0.3475])
    elif name == 'ImageNet':
        # RGB
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif name == 'EuroSAT':
        # RGB
        mean = [0.344, 0.380, 0.408]
        std = [0.203, 0.137, 0.116]
    elif name == 'Google-China':
        # RGB
        mean = [0.354, 0.379, 0.366]
        std = [0.177, 0.165, 0.165]
    elif name == 'RSSCN7':
        # RGB
        mean = [0.373, 0.384, 0.343]
        std = [0.199, 0.180, 0.183]
    elif name == 'RS19-WHU':
        # RGB
        mean = [0.426, 0.448, 0.402]
        std = [0.242, 0.219, 0.229]
    elif name == 'AID':
        # RGB
        mean = [0.398, 0.409, 0.369]
        std = [0.217, 0.194, 0.192]
    elif name == 'UCMerced_LandUse':
        # RGB
        mean = [0.484, 0.49, 0.451]
        std = [0.218, 0.202, 0.196]
    elif name == 'RSD46-WHU':
        # RGB
        mean = [0.392, 0.417, 0.391]
        std = [0.241, 0.225, 0.226]
        # Single Channel
        # mean = []
        # std = []
    elif name == 'PatternNet':
        # RGB
        mean = [0.359, 0.36, 0.319]
        std = [0.196, 0.185, 0.179]
    elif name == 'RSI-CB128':
        # RGB
        mean = [0.411, 0.418, 0.379]
        std = [0.209, 0.194, 0.197]
    elif name == 'RSI-CB256':
        # RGB
        mean = [0.424, 0.432, 0.393]
        std = [0.237, 0.218, 0.217]
    else:
        raise ValueError()
    return mean, std


def create_transform(config: yacs.config.CfgNode, is_train: bool) -> Callable:
    if config.model.type == 'cifar':
        return create_cifar_transform(config, is_train)
    elif config.model.type == 'imagenet':
        return create_imagenet_transform(config, is_train)
    elif config.model.type == 'custom':
        return create_custom_transform(config, is_train)
    else:
        raise ValueError


def create_cifar_transform(config: yacs.config.CfgNode,
                           is_train: bool) -> Callable:
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_random_crop:
            transforms.append(RandomCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))

        transforms.append(Normalize(mean, std))

        if config.augmentation.use_cutout:
            transforms.append(Cutout(config))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.use_dual_cutout:
            transforms.append(DualCutout(config))

        transforms.append(ToTensor())
    else:
        transforms = [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)


def create_imagenet_transform(config: yacs.config.CfgNode,
                              is_train: bool) -> Callable:
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizeCrop(config))
        else:
            transforms.append(CenterCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))

        transforms.append(Normalize(mean, std))

        if config.augmentation.use_cutout:
            transforms.append(Cutout(config))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.use_dual_cutout:
            transforms.append(DualCutout(config))

        transforms.append(ToTensor())
    else:
        transforms = []
        if config.tta.use_resize:
            transforms.append(Resize(config))
        if config.tta.use_center_crop:
            transforms.append(CenterCrop(config))
        transforms += [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)


def create_custom_transform(config: yacs.config.CfgNode,
                              is_train: bool) -> Callable:
    mean, std = _get_dataset_stats(config)
    if is_train:
        transforms = []
        if config.augmentation.use_random_crop:
            transforms.append(RandomResizeCrop(config))
        else:
            transforms.append(CenterCrop(config))
        if config.augmentation.use_random_horizontal_flip:
            transforms.append(RandomHorizontalFlip(config))

        transforms.append(Normalize(mean, std))

        if config.augmentation.use_cutout:
            transforms.append(Cutout(config))
        if config.augmentation.use_random_erasing:
            transforms.append(RandomErasing(config))
        if config.augmentation.use_dual_cutout:
            transforms.append(DualCutout(config))

        transforms.append(ToTensor())
    else:
        transforms = []
        if config.tta.use_resize:
            transforms.append(Resize(config))
        if config.tta.use_center_crop:
            transforms.append(CenterCrop(config))
        transforms += [
            Normalize(mean, std),
            ToTensor(),
        ]

    return torchvision.transforms.Compose(transforms)
