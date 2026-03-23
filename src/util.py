import argparse
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def deep_merge(base, override):
    result = {**base}

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def read_configs(base_path, specific_path):
    base_config = read_config(base_path)
    specific_config = read_config(specific_path)

    return deep_merge(base_config, specific_config)


def parse_args_and_get_config(stage):
    base_config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml' 
    parser = argparse.ArgumentParser()

    if stage == 'train':
        parser.add_argument('--config', required=True, help='Path to configuration file')
        args = parser.parse_args()
        return read_configs(base_config_path, args.config)
    elif stage == 'evaluate':
        parser.add_argument('--configs', nargs="+", required=True, help='Paths to model files')
        args = parser.parse_args()
        return [read_configs(base_config_path, c) for c in args.configs]
    else:
        raise ValueError("Unknown stage. Only 'train' and 'evaluate' supported so far")


def plot_precision_recall_curve(recalls, precisions, labels, auprcs):
    plt.figure(figsize=(10, 7))

    for i in range(len(recalls)):
        plt.plot(recalls[i], precisions[i], label=f'{labels[i]} (AUPRC = {auprcs[i]:.3f})')

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve for Chest X-Ray Images', fontsize=16)
    plt.legend(loc='lower left')

    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/PRAUC.png')


def plot_learning_curve(training_cost, eval_cost, model_name):
    plt.figure(figsize=(6, 6))
    plt.plot(training_cost)
    plt.plot(eval_cost)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve')
    plt.legend(['training', 'eval'])

    os.makedirs('outputs', exist_ok=True)
    plt.savefig(f'outputs/learning_curve_{model_name}.png')

    
def plot_confusion_matrix(y_test, predictions, model_name):
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot()

    plt.savefig(f'outputs/{model_name}_confusion_matrix.png')


def plot_grad_cams(model_name, grad_cams):
    os.makedirs(f'outputs/grad_cam/{model_name}', exist_ok=True)

    if model_name == 'resnet50':
        model_cmap = None
    else:
        model_cmap = 'gray'

    for i in range(len(grad_cams)):
        _, ax = plt.subplots(1, 2, figsize=(6, 6))
        label, y_pred, img_np, upsampled_heatmap = grad_cams[i]
        ax[0].imshow(img_np, cmap=model_cmap)
        ax[0].axis('off')
        ax[0].set_title('PNEUMONIA' if label == 1 else 'NORMAL')
        ax[1].imshow(img_np, cmap=model_cmap)
        ax[1].imshow(upsampled_heatmap, alpha=.5, cmap='coolwarm')
        ax[1].axis('off')
        ax[1].set_title('PNEUMONIA' if y_pred == 1 else 'NORMAL')

        plt.savefig(f'outputs/grad_cam/{model_name}/{i}.png')
        plt.close()


def find_best_threshold(y_true, y_pred_proba, beta):
    precision, recall, threshold = precision_recall_curve(y_true, y_pred_proba)
    f1 = (1 + beta**2) * precision[:-1] * recall [:-1] / (beta**2 * precision[:-1] + recall[:-1] + 1e-8)
    best_threshold_idx = np.argmax(f1)
    
    return threshold[best_threshold_idx]


def setup_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')