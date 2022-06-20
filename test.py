#
#  test.py
#  ./
#
#  Created by Joshua Wen on 2022/06/20.
#  Copyright © 2022 Joshua Wen. All rights reserved.
#
import argparse
import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import tqdm


from pytorch_image_classification import (
    apply_data_parallel_wrapper,
    create_loss,
    create_model,
    get_default_config,
    update_config,
    create_transform,
)
from pytorch_image_classification.utils import (
    AverageMeter,
    create_logger,
    get_rank,
    SklearnTools,
)


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config)
    config.merge_from_list(args.options)
    update_config(config)
    config.freeze()
    return config


def evaluate(config, model, test_dataset, test_loader, loss_func, logger):
    device = torch.device(config.device)
    model.to(device)

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()

    pred_raw_all = []
    pred_prob_all = []
    pred_label_all = []
    gt_labels = []
    with torch.no_grad():
        for data, targets in tqdm.tqdm(test_loader):
            data = data.to(device)
            targets = targets.to(device)

            outputs = model(data)
            loss = loss_func(outputs, targets)

            pred_raw_all.append(outputs.cpu().numpy())
            pred_prob_all.append(F.softmax(outputs, dim=1).cpu().numpy())

            _, preds = torch.max(outputs, dim=1)
            pred_label_all.append(preds.cpu().numpy())

            gt_labels.append(targets.cpu().numpy())

            loss_ = loss.item()
            correct_ = preds.eq(targets).sum().item()
            num = data.size(0)

            loss_meter.update(loss_, num)
            correct_meter.update(correct_, 1)

        accuracy = correct_meter.sum / len(test_loader.dataset)

        elapsed = time.time() - start
        logger.info(f'Elapsed {elapsed:.2f}')
        logger.info(f'Loss {loss_meter.avg:.4f} Accuracy {accuracy:.4f}')

    preds = np.concatenate(pred_raw_all)
    probs = np.concatenate(pred_prob_all)

    predicted_labels = np.concatenate(pred_label_all)
    gt_labels = np.concatenate(gt_labels)

    tools = SklearnTools(test_dataset, gt_labels, predicted_labels)
    tools.plot_confusion_matrix(config)
    tools.plot_roc_curve(config, probs)
    logger.info(tools.get_classification_report())

    return preds, probs, predicted_labels, loss_meter.avg, accuracy


def main():
    config = load_config()

    if config.test.output_dir is None:
        output_dir = pathlib.Path(config.test.checkpoint).parent
    else:
        output_dir = pathlib.Path(config.test.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger = create_logger(name=__name__, distributed_rank=get_rank())

    model = create_model(config)
    model = apply_data_parallel_wrapper(config, model)
    checkpoint = torch.load(config.test.checkpoint)
    model.load_state_dict(checkpoint['model'])
    test_transform = create_transform(config, is_train=False)
    test_dataset = torchvision.datasets.ImageFolder(
        config.dataset.dataset_dir + '/train',
        transform=test_transform)
    print(test_dataset.class_to_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test.batch_size,
        num_workers=config.test.dataloader.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=config.test.dataloader.pin_memory)
    _, test_loss = create_loss(config)

    preds, probs, labels, loss, acc = evaluate(config, model, test_dataset, test_loader,
                                               test_loss, logger)

    output_path = output_dir / f'predictions.npz'
    np.savez(output_path,
             preds=preds,
             probs=probs,
             labels=labels,
             loss=loss,
             acc=acc)


if __name__ == '__main__':
    main()
