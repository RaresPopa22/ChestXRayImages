import logging
import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, roc_auc_score, \
    f1_score

from src.data_processing import get_test_data
from src.model import BaseCNN
from src.util import parse_args_and_get_config, plot_confusion_matrix, plot_precision_recall_curve, find_best_threshold, setup_device


logger = logging.getLogger(__name__)
device = setup_device()


def load_mean_and_std(config, grayscale):
    if grayscale:
        path = config['data_paths']['processed']['mean_and_std_gray']
    else:
        path = config['data_paths']['processed']['mean_and_std_color']

    mean_and_std_dict = torch.load(path)
    
    return mean_and_std_dict.get('mean'), mean_and_std_dict.get('std')


def load_model(config, model_name):
    model_path = Path(config['model_output_paths']['model'])

    if model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(2048, 1)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
    elif model_name == 'base_cnn':
        model = BaseCNN(config)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(device)
    else:
        raise ValueError(f'Unknown model requested {model_path.stem}')
    
    model.eval()
    return model


def get_predictions_and_labels(model, test_data):
    predictions = []
    labels_Y = []

    with torch.no_grad():
        for batch_X, batch_Y in test_data:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device).unsqueeze(1).float()

            hypothesis = model(batch_X)
            predictions.append(torch.sigmoid(hypothesis))
            labels_Y.append(batch_Y)

    y_test = torch.cat(labels_Y).cpu().numpy()
    y_pred_proba = torch.cat(predictions).cpu().numpy()
    
    return y_test, y_pred_proba


def print_classification_report(y_test, y_pred, model_name):
    class_labels = ['NORMAL', 'PNEUMONIA']
    report = classification_report(y_test, y_pred, target_names=[class_labels[0], class_labels[1]])
    logger.info(f'Classification report for {model_name}')
    logger.info(f'\n{report}')


def evaluate(configs):
    results, precisions, recalls, labels, auprcs = [], [], [], [], []

    for config in configs:
        model_path = Path(config['model_output_paths']['model'])

        grayscale = model_path.stem != 'resnet50'
        mean, std = load_mean_and_std(config, grayscale)
        test_data = get_test_data(config, mean, std, grayscale=grayscale)

        model = load_model(config, model_path.stem)
        y_test, y_pred_proba = get_predictions_and_labels(model, test_data)

        optimal_threshold = find_best_threshold(y_test, y_pred_proba, config['f1_score']['beta'])
        logger.info(f'Using optimal threshold={optimal_threshold}')
        
        y_pred = (y_pred_proba > optimal_threshold).astype('int32')
        print_classification_report(y_test, y_pred, model_path.stem)

        auprc = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        results.append({
            'Model': model_path.stem,
            'AUPRC': auprc,
            'ROC AUC': roc_auc,
            'F1-Score (PNEUMONIA)': f1
        })

        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        recalls.append(recall)
        precisions.append(precision)
        labels.append(model_path.stem)
        auprcs.append(auprc)
        plot_confusion_matrix(y_test, y_pred, model_path.stem)

    results_df = pd.DataFrame(results).set_index('Model')
    logger.info(f'\n{results_df.to_string()}')

    plot_precision_recall_curve(recalls, precisions, labels, auprcs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    configs = parse_args_and_get_config('evaluate')
    evaluate(configs)
