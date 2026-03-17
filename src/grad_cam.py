import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from src.data_processing import get_test_data
from src.evaluate import load_mean_and_std, load_model
from src.util import parse_args_and_get_config, plot_grad_cams, setup_device


device = setup_device()
logger = logging.getLogger(__name__)
activations = []
gradients = []


def collect_idxs(total_count, gc_dataset, target):
    labels = gc_dataset.targets
    return np.where(np.array(labels) == target)[0][:total_count]


def get_dataset_idxs(total_examples, dataset):
    pos_idxs = collect_idxs(total_examples, dataset, 1)
    neg_idxs = collect_idxs(total_examples, dataset, 0)
    return np.concatenate((pos_idxs, neg_idxs))


def get_prediction(config, score):
    y_proba = torch.sigmoid(score)
    optimal_threshold = config['optimal_threshold']
    return (y_proba > optimal_threshold).item()


def save_activations(module, input, output):
    activations.append(output.detach().cpu().numpy().squeeze())


def save_gradients(module, input, output):
    gradients.append(output[0].cpu().numpy().squeeze())


def register_hooks(model, model_name):
    if model_name == 'resnet50':
        activation_hook = model.layer4.register_forward_hook(save_activations)
        grad_hook = model.layer4.register_full_backward_hook(save_gradients)
    elif model_name == 'base_cnn':
        activation_hook = model.layer_3_c.register_forward_hook(save_activations)
        grad_hook = model.layer_3_c.register_full_backward_hook(save_gradients)
    else:
        raise ValueError(f'Unknown model requested {model_name}')
    
    return activation_hook, grad_hook


def denormalize(img, mean, std, model_name):
    mean = mean.to(device)
    std = std.to(device)

    if model_name == 'resnet50':
        mean = mean[:, None, None]
        std = std[:, None, None]
    
    return img * std + mean


def run_grad_cam(configs):
    for config in configs:
        model_path = Path(config['model_output_paths']['model'])
        total_examples = config['hyperparameters']['grad_cam_example_size']

        grayscale = model_path.stem != 'resnet50'
        mean, std = load_mean_and_std(config, grayscale)
        gc_dataset = get_test_data(config, mean, std, grayscale=grayscale).dataset

        model = load_model(config, model_path.stem)
        activation_hook, grad_hook = register_hooks(model, model_path.stem)

        grad_cams = []

        for i in get_dataset_idxs(total_examples, gc_dataset):
            activations.clear()
            gradients.clear()

            img, label = gc_dataset[i]
            img = img.to(device)
            output = model(img[None, :, :, :])
            score = output[0]
            score.backward()

            gradients_aggregated = np.mean(gradients[0], axis=(1, 2))
            weighted_activations = np.sum(activations[0] * gradients_aggregated[:, np.newaxis, np.newaxis], axis=0)
            relu_weighted_activations = np.maximum(weighted_activations, 0)
            upsampled_heatmap = cv2.resize(relu_weighted_activations, (img.size(2), img.size(1)), interpolation=cv2.INTER_LINEAR)

            img = denormalize(img, mean, std, model_path.stem)
            img_np = np.transpose(img.detach().cpu().numpy(), axes=[1, 2, 0])

            y_pred = get_prediction(config, score)
            grad_cams.append((label, y_pred, img_np, upsampled_heatmap))

            model.zero_grad()

        activation_hook.remove()
        grad_hook.remove()
        
        plot_grad_cams(model_path.stem, grad_cams)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    configs = parse_args_and_get_config('evaluate')

    run_grad_cam(configs)