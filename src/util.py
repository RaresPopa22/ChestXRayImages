import argparse

import yaml
from matplotlib import pyplot as plt


def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_configs(base_path, specific_path):
    base_config = read_config(base_path)
    specific_config = read_config(specific_path)

    return {**base_config, **specific_config}


def parse_args_and_get_config(stage):
    base_config = "../config/base_config.yaml"
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