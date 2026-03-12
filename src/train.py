import logging

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
from src.data_processing import load_training_data
from src.model import BaseCNN
from src.util import parse_args_and_get_config, plot_learning_curve
from contextlib import nullcontext


logger = logging.getLogger(__name__)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def log_metrics(epoch, training_acc, training_cost, eval_acc, eval_cost):
    logger.info(f'Epoch={epoch+1} Training: cost={training_cost.item():.2f} accuracy={training_acc.item():.2f} Eval: cost={eval_cost.item():.2f} accuracy={eval_acc.item():.2f}')


def freeze_all_except_finetune_layers(model, hyperparam_config):
    for name, layer in model.named_children():
        if name not in hyperparam_config['finetune_layers']:
            for params in layer.parameters():
                params.requires_grad = False


def get_model(config, model_name):
    hyperparam_config = config['hyperparameters']
    target_size = config['data_paths']['raw_data']['target_size']

    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        freeze_all_except_finetune_layers(model, hyperparam_config)
        model.fc = nn.Linear(2048, 1)
    elif model_name == 'base_cnn':
        model = BaseCNN(target_size)
    else:
        raise ValueError(f'Unknown model requested: {model_name}')

    return model.to(device)


def get_optimizer(model, current_lr, layer4_lr):
    if layer4_lr is not None:
        return torch.optim.Adam([
            {'params': model.fc.parameters(), 'lr': current_lr},
            {'params': model.layer4.parameters(), 'lr': layer4_lr},
        ])
    else:
        return torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)


def get_lr_scheduler(hyperparam_config, optimizer):
    return ReduceLROnPlateau(
        optimizer, 
        patience=hyperparam_config['lr_scheduler_patience'], 
        threshold=hyperparam_config['lr_threshold']
        )


def maybe_nograd(use_ng):
    if use_ng:
        return torch.no_grad()
    else:
        return nullcontext()


def maybe_autocast(device, use_amp):
    if use_amp:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    else:
        return nullcontext()


def run_batch_for(data_set, model, criterion, optimizer, scaler, training=True):
    avg_cost = torch.tensor(0.0, device=device)
    avg_accuracy = torch.tensor(0.0, device=device)

    if training:
        model.train()
    else:
        model.eval()

    with maybe_nograd(not training):
        for batch_X, batch_Y in data_set:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device).unsqueeze(1).float()

            with maybe_autocast(device, training):
                output = model(batch_X)
                loss = criterion(output, batch_Y)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            prediction = (output.data > 0).int()
            accuracy = (prediction == batch_Y.data).sum() / len(prediction)
            avg_accuracy += accuracy / len(data_set)
            avg_cost += loss.data / len(data_set)

    return avg_cost, avg_accuracy


def drop_lr_if_needed(optimizer, current_lr, layer4_lr=None):
    if current_lr != optimizer.param_groups[0]['lr']:
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Dropping learning rate to: {current_lr}')
    if layer4_lr is not None and layer4_lr != optimizer.param_groups[1]['lr']:
        layer4_lr = optimizer.param_groups[1]['lr']
        logger.info(f'Dropping learning rate for layer 4 to: {layer4_lr}')
    
    return current_lr, layer4_lr


def train(config):
    model_name = config['model_name']
    hyperparam_config = config['hyperparameters']

    model = get_model(config, model_name)
    training_set, eval_set, imbalance_ratio = load_training_data(config, model_name != 'resnet50')

    current_lr = hyperparam_config['learning_rate']
    layer4_lr = hyperparam_config['learning_rate_layer4'] if model_name == 'resnet50' else None

    optimizer = get_optimizer(model, current_lr, layer4_lr)
    lr_scheduler = get_lr_scheduler(hyperparam_config, optimizer)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(imbalance_ratio, device=device))

    plt_training_cost = []
    plt_eval_cost = []
    best_eval = float('inf')

    patience = 0
    scaler = GradScaler(device=device.type)
    
    for epoch in range(hyperparam_config['epochs']):
        if patience > hyperparam_config['patience']:
            break

        training_avg_cost, training_accuracy = run_batch_for(training_set, model, criterion, optimizer, scaler, training=True)
        eval_avg_cost, eval_accuracy = run_batch_for(eval_set, model, criterion, optimizer, scaler, training=False)

        if best_eval > (eval_avg_cost + 1e-4):
            best_eval = eval_avg_cost
            torch.save(model.state_dict(), config['model_output_paths']['model'])
        else:
            patience += 1

        log_metrics(epoch, training_accuracy, training_avg_cost, eval_accuracy, eval_avg_cost)
        plt_training_cost.append(training_avg_cost.item())
        plt_eval_cost.append(eval_avg_cost.item())

        lr_scheduler.step(eval_avg_cost)
        current_lr, layer4_lr = drop_lr_if_needed(optimizer, current_lr, layer4_lr)
    
    plot_learning_curve(plt_training_cost, plt_eval_cost, model_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    config = parse_args_and_get_config('train')
    torch.manual_seed(config['seed'])
    train(config)
