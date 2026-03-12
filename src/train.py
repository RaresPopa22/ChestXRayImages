import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
from src.data_processing import compute_mean_and_std, get_data_and_idxs, get_train_val_data
from src.model import CNN, BaseCNN
from src.util import parse_args_and_get_config
from collections import Counter


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def log_metrics(epoch, training_acc, training_cost, eval_acc, eval_cost):
    logger.info(f'Epoch={epoch+1} Training: cost={training_cost.item():.2f} accuracy={training_acc.item():.2f} Eval: cost={eval_cost.item():.2f} accuracy={eval_acc.item():.2f}')


def plot_learning_curve(training_cost, eval_cost, model_name):
    plt.figure(figsize=(6, 6))
    plt.plot(training_cost)
    plt.plot(eval_cost)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve')
    plt.legend(['training', 'eval'])
    plt.savefig(f'learning_curve_{model_name}.jpg')


def train(config):
    model_name = config['model_name']
    hyperparam_config = config['hyperparameters']
    target_size = config['data_paths']['raw_data']['target_size']
    data, train_idxs, val_idxs = get_data_and_idxs(config)

    try:
        grayscale = model_name != 'resnet50'
        path = config['data_paths']['processed']['mean_and_std_gray'] if grayscale else config['data_paths']['processed']['mean_and_std_color']
        mean_and_std_dict = torch.load(path)
        mean = mean_and_std_dict.get('mean')
        std = mean_and_std_dict.get('std')
        logger.info(f'Using existing mean={mean} and std={std}')
    except FileNotFoundError:
        mean, std = compute_mean_and_std(data, train_idxs, config, model_name != 'resnet50')

    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        for name, layer in model.named_children():
            if name not in hyperparam_config['finetune_layers']:
                for params in layer.parameters():
                    params.requires_grad = False
        
        model.fc = nn.Linear(2048, 1)
        training_set, eval_set = get_train_val_data(config, data, train_idxs, val_idxs, mean, std, False)
    elif model_name == 'base_cnn':
        model = BaseCNN(target_size)
        training_set, eval_set = get_train_val_data(config, data, train_idxs, val_idxs, mean, std)
    else:
        raise ValueError(f'Unknown model requested: {model_name}')
    
    model.to(device)
    training_batch = len(training_set)
    eval_batch = len(eval_set)

    current_lr = hyperparam_config['learning_rate']

    if model_name == 'resnet50':
        layer4_lr = hyperparam_config['learning_rate_layer4']
        optimizer = torch.optim.Adam([
            {'params': model.fc.parameters(), 'lr': current_lr},
            {'params': model.layer4.parameters(), 'lr': layer4_lr},
        ])
    else:
        optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
    
    lr_scheduler = ReduceLROnPlateau(
        optimizer, 
        patience=hyperparam_config['lr_scheduler_patience'], 
        threshold=hyperparam_config['lr_threshold']
        )
    
    all_data = training_set.dataset.subset.dataset.targets
    training_data = [all_data[i] for i in train_idxs]
    label_counts = Counter(training_data)
    imbalance_ratio = label_counts[0] / label_counts[1]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(imbalance_ratio, device=device))

    plt_training_cost = []
    plt_eval_cost = []

    best_eval = float('inf')
    patience = 0

    scaler = GradScaler(device=device.type)

    model.train()
    
    for epoch in range(hyperparam_config['epochs']):
        training_avg_cost = torch.tensor(0.0, device=device)
        training_accuracy = torch.tensor(0.0, device=device)
        eval_avg_cost = torch.tensor(0.0, device=device)
        eval_accuracy = torch.tensor(0.0, device=device)

        if patience > hyperparam_config['patience']:
            break

        for batch_X, batch_Y in training_set:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device).unsqueeze(1).float()

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, dtype=torch.float16):
                output = model(batch_X)
                loss = criterion(output, batch_Y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            prediction = (output.data > 0).int()
            accuracy = (prediction == batch_Y.data).sum() / len(prediction)
            training_accuracy += accuracy / training_batch
            training_avg_cost += loss.data / training_batch

        model.eval()
        with torch.no_grad():
            for batch_X, batch_Y in eval_set:
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device).unsqueeze(1).float()

                hypothesis = model(batch_X)
                cost = criterion(hypothesis, batch_Y)

                prediction = (hypothesis.data > 0).int()
                accuracy = (prediction == batch_Y.data).sum() / len(prediction)
                eval_accuracy += accuracy / eval_batch
                eval_avg_cost += cost.data / eval_batch

        if best_eval > (eval_avg_cost + 1e-4):
            best_eval = eval_avg_cost
            torch.save(model.state_dict(), config['model_output_paths']['model'])
        else:
            patience += 1

        log_metrics(epoch, training_accuracy, training_avg_cost, eval_accuracy, eval_avg_cost)
        plt_training_cost.append(training_avg_cost.item())
        plt_eval_cost.append(eval_avg_cost.item())

        lr_scheduler.step(eval_avg_cost)

        if current_lr != optimizer.param_groups[0]['lr']:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Dropping learning rate to: {current_lr}')

        if model_name == 'resnet50' and layer4_lr != optimizer.param_groups[1]['lr']:
            layer4_lr = optimizer.param_groups[1]['lr']
            logger.info(f'Dropping learning rate for layer 4 to: {layer4_lr}')


        model.train()
    
    plot_learning_curve(plt_training_cost, plt_eval_cost, model_name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    config = parse_args_and_get_config('train')
    torch.manual_seed(config['seed'])
    train(config)
