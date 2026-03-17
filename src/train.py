import logging

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50, ResNet50_Weights
from src.data_processing import load_training_data
from src.model import BaseCNN
from src.util import parse_args_and_get_config, plot_learning_curve, setup_device
from contextlib import nullcontext


logger = logging.getLogger(__name__)
device = setup_device()


def log_metrics(epoch, training_acc, training_cost, eval_acc, eval_cost):
    logger.info(f'Epoch={epoch+1} Training: cost={training_cost.item():.2f} accuracy={training_acc.item():.2f} Eval: cost={eval_cost.item():.2f} accuracy={eval_acc.item():.2f}')


def freeze_all_except_finetune_layers(model, hyperparam_config):
    for name, layer in model.named_children():
        if name not in hyperparam_config['finetune_layers']:
            for params in layer.parameters():
                params.requires_grad = False


def get_model(config, model_name):
    hyperparam_config = config['hyperparameters']

    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        freeze_all_except_finetune_layers(model, hyperparam_config)
        model.fc = nn.Linear(2048, 1)
    elif model_name == 'base_cnn':
        model = BaseCNN(config)
    else:
        raise ValueError(f'Unknown model requested: {model_name}')

    return model.to(device)


def get_optimizer(hyperparam_config, model, model_name):
    current_lr = hyperparam_config['learning_rate']

    if model_name == 'resnet50':
        return torch.optim.Adam([
            {'params': model.fc.parameters(), 'lr': current_lr, 'weight_decay': hyperparam_config['weight_decay']},
            {'params': model.layer4.parameters(), 'lr': hyperparam_config['learning_rate_layer4'], 'weight_decay': hyperparam_config['weight_decay']},
        ])
    else:
        return torch.optim.Adam(
            params = filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr, weight_decay=hyperparam_config['weight_decay']
        )


def get_lr_scheduler(hyperparam_config, optimizer):
    lr_config = hyperparam_config['lr_scheduler']
    return CosineAnnealingLR(
        optimizer,
        T_max=lr_config['T_max'],
        eta_min=lr_config['eta_min']
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


def run_batch_for(data_set, model, criterion, optimizer=None, scaler=None, use_amp=False):
    avg_cost = torch.tensor(0.0, device=device)
    avg_accuracy = torch.tensor(0.0, device=device)
    is_training = optimizer is not None and scaler is not None

    if is_training:
        model.train()
    else:
        model.eval()

    with maybe_nograd(not is_training):
        for batch_X, batch_Y in data_set:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device).unsqueeze(1).float()

            with maybe_autocast(device, use_amp):
                output = model(batch_X)
                loss = criterion(output, batch_Y)

            if is_training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            prediction = (output.data > 0).int()
            accuracy = (prediction == batch_Y.data).sum() / len(prediction)
            avg_accuracy += accuracy / len(data_set)
            avg_cost += loss.data / len(data_set)

    return avg_cost, avg_accuracy


def train(config):
    model_name = config['model_name']
    hyperparam_config = config['hyperparameters']

    model = get_model(config, model_name)
    training_set, eval_set, imbalance_ratio = load_training_data(config, model_name != 'resnet50')

    optimizer = get_optimizer(hyperparam_config, model, model_name)
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

        training_avg_cost, training_accuracy = run_batch_for(
            training_set, model, criterion, optimizer, scaler, use_amp=hyperparam_config['use_amp']
            )
        eval_avg_cost, eval_accuracy = run_batch_for(
            eval_set, model, criterion, use_amp=hyperparam_config['use_amp']
            )

        if best_eval > (eval_avg_cost + 1e-4):
            best_eval = eval_avg_cost
            torch.save(model.state_dict(), config['model_output_paths']['model'])
        else:
            patience += 1

        log_metrics(epoch, training_accuracy, training_avg_cost, eval_accuracy, eval_avg_cost)
        plt_training_cost.append(training_avg_cost.item())
        plt_eval_cost.append(eval_avg_cost.item())

        lr_scheduler.step()
    
    plot_learning_curve(plt_training_cost, plt_eval_cost, model_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    config = parse_args_and_get_config('train')
    torch.manual_seed(config['seed'])
    train(config)
