import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score


def compute_accuracy(config, outputs, targets, augmentation, topk=(1, )):
    if augmentation:
        if config.augmentation.use_mixup or config.augmentation.use_cutmix:
            targets1, targets2, lam = targets
            accs1 = accuracy(outputs, targets1, topk)
            accs2 = accuracy(outputs, targets2, topk)
            accs = tuple([
                lam * acc1 + (1 - lam) * acc2
                for acc1, acc2 in zip(accs1, accs2)
            ])
        elif config.augmentation.use_ricap:
            weights = []
            accs_all = []
            for labels, weight in zip(*targets):
                weights.append(weight)
                accs_all.append(accuracy(outputs, labels, topk))
            accs = []
            for i in range(len(accs_all[0])):
                acc = 0
                for weight, accs_list in zip(weights, accs_all):
                    acc += weight * accs_list[i]
                accs.append(acc)
            accs = tuple(accs)
        elif config.augmentation.use_dual_cutout:
            outputs1, outputs2 = outputs[:, 0], outputs[:, 1]
            accs = accuracy((outputs1 + outputs2) / 2, targets, topk)
        else:
            accs = accuracy(outputs, targets, topk)
    else:
        accs = accuracy(outputs, targets, topk)
    return accs


def accuracy(outputs, targets, topk=(1, )):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1 / batch_size))
    return res


#
#  metrics.py
#  utils
#
#  Updated by Joshua Wen on 2022/06/20.
#  Copyright © 2022 Joshua Wen. All rights reserved.
#
class SklearnTools:
    def __init__(self, dataset, gt_labels, pred_labels):
        self.dataset = dataset
        self.class_names = self.dataset.class_to_idx
        self.class_number = len(self.class_names)
        self.gt_labels = gt_labels
        self.pred_labels = pred_labels
        self.colors = []
        self.gt_index = []
        self.gt_index_and_class = []

    def plot_confusion_matrix(self, config):
        for k, v in self.class_names.items():
            self.gt_index.append(str(v + 1))
            self.gt_index_and_class.append(str(v + 1) + '-' + k)
        matrix = confusion_matrix(self.gt_labels, self.pred_labels)
        plt.matshow(matrix)
        plt.colorbar()
        plt.title('Confusion Matrix (Dataset: %s | Model: %s)' % (config.dataset.name, config.model.name))
        plt.xlabel('Prediction')
        plt.ylabel('Ground Truth')
        plt.xticks(np.arange(matrix.shape[1]), self.gt_index)
        plt.yticks(np.arange(matrix.shape[1]), self.gt_index_and_class)
        plt.show()

    def plot_roc_curve(self, config, probs):
        for i in range(self.class_number):
            self.colors.append(random_color())
        gt_labels_one_hot = label_binarize(self.gt_labels, classes=np.arange(self.class_number))
        fpr, tpr, roc_auc = dict(), dict(), dict()
        plt.figure(figsize=(8, 8))
        for i in range(self.class_number):
            fpr[i], tpr[i], threshold = roc_curve(gt_labels_one_hot[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

            plt.plot(fpr[i], tpr[i], color=self.colors[i],
                     label='%s (area = %0.4f)' % (self.gt_index_and_class[i], roc_auc[i]))
        score = self.get_roc_auc_score(gt_labels_one_hot, probs)
        lw = 2
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('ROC Curve (Dataset: %s | Model: %s | AUC=%.4f)' % (config.dataset.name, config.model.name, score))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.show()

    def get_classification_report(self, config, digits=4, is_excel=True):
        report = classification_report(self.gt_labels, self.pred_labels,
                                       target_names=self.class_names.keys(), digits=digits)
        if is_excel:
            report_dict = classification_report(self.gt_labels, self.pred_labels,
                                                output_dict=True, target_names=self.class_names.keys(), digits=digits)
            report_to_excel(report_dict, config)
        return "Classification Report: \n%s" % report

    @staticmethod
    def get_roc_auc_score(gt_labels_one_hot, probs, multi_class='ovr'):
        return roc_auc_score(gt_labels_one_hot, probs, multi_class=multi_class)


def random_color():
    import random
    color_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'A', 'B', 'C', 'D', 'E', 'F']
    color = ''
    for i in range(6):
        color += color_arr[random.randint(1, 14)]
    return '#' + color


def report_to_excel(report, config):
    import pandas
    df = pandas.DataFrame(report).transpose()
    df.to_excel(config.test.output_dir + '/%s+%s.xlsx' % (config.dataset.name, config.model.name), sheet_name='Sheet1')
