import argparse
import json
import logging
from pathlib import Path

import torch

from src.data_processing import get_predict_data
from src.evaluate import load_mean_and_std, load_model
from src.model import BaseCNN
from src.util import read_config, read_configs, setup_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = setup_device()


def predict(config, input_path):
    model_name = config['model_name']
    grayscale = model_name != 'resnet50'
    mean, std = load_mean_and_std(config, grayscale)
    data = get_predict_data(config, input_path, mean, std).to(device)
    model = load_model(config, model_name)
    hypothesis = model(data)

    return torch.sigmoid(hypothesis).item()



if __name__ == '__main__':
    base_config_path = Path(__file__).parent.parent / 'config' / 'base_config.yaml' 
    parser = argparse.ArgumentParser(description='Run chest X-ray prediction on new data')
    parser.add_argument('--config', required=True, help='Path to configuration file')
    parser.add_argument('--input', required=True, help='Path to input')
    parser.add_argument('--output', required=True, help='Path to save prediction in json format')
    args = parser.parse_args()

    hypothesis = predict(read_configs(base_config_path, args.config), args.input)
    with open(args.output, 'w') as f:
        json.dump(f'hypothesis: {hypothesis}', f)
            
