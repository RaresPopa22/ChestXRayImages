import argparse

import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_configs(base_path, specific_path):
    base_config = read_config(base_path)
    specific_config = read_config(specific_path)

    return {**base_config, **specific_config}


def parse_args_and_get_config(stage):
    base_config = "config/base_config.yaml"
    parser = argparse.ArgumentParser()

    if stage == 'train':
        parser.add_argument('--config', required=True, help='Path to configuration file')
        args = parser.parse_args()
        return read_configs(base_config, args.config)
    elif stage == 'evaluate':
        parser.add_argument('--config', required=True, help='Path to configuration file')
        parser.add_argument('--models', nargs="+", required=True, help='Paths to model files')
        args = parser.parse_args()
        return read_configs(base_config, args.config), args.models
    else:
        raise ValueError("Unknown stage. Only 'train' and 'evaluate' supported so far")


def plot_precision_recall_curve(recalls, precisions, labels, auprcs):
    fig, ax = plt.subplots(figsize=(10, 7))

    for i in range(len(recalls)):
        ax.plot(recalls[i], precisions[i], label=f'{labels[i]} (AUPRC = {auprcs[i]:.3f})')

    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve for Chest X-Ray Images', fontsize=16)
    ax.legend(loc='lower left')
    plt.show()


def find_best_threshold(y_true, y_pred_proba):
    best_threshold = 0
    best_f1 = 0
    thresholds = np.arange(0.01, 1.0, 0.01)

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        current_f1 = f1_score(y_true, y_pred)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    print(f'Best F1-Score: {best_f1:.4f} found at threshold: {best_threshold:.2f}')
    return best_threshold