import argparse

import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_recall_curve


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
    base_config_path = "config/base_config.yaml"
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
    plt.savefig('PRAUC.jpg')


def find_best_threshold(y_true, y_pred_proba, beta):
    precision, recall, threshold = precision_recall_curve(y_true, y_pred_proba)
    f1_score = (1 + beta**2) * precision[:-1] * recall [:-1] / (beta**2 * precision[:-1] + recall[:-1] + 1e-8)
    best_threshold_idx = np.argmax(f1_score)
    
    return threshold[best_threshold_idx]