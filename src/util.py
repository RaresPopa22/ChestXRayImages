import argparse

import yaml


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
        return read_configs(base_config, args.config)
    else:
        raise ValueError("Unknown stage. Only 'train' and 'evaluate' supported so far")

