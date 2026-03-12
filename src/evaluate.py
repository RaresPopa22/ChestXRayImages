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
from src.util import parse_args_and_get_config, plot_precision_recall_curve, find_best_threshold

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def evaluate(configs):
    results, precisions, recalls, labels, auprcs = [], [], [], [], []

    for config in configs:
        model_path = Path(config['model_output_paths']['model'])

        grayscale = model_path.stem != 'resnet50'
        gray_statistics_path = config['data_paths']['processed']['mean_and_std_gray']
        rgb_statistics_path = config['data_paths']['processed']['mean_and_std_color']
        statistics_path = gray_statistics_path if grayscale else rgb_statistics_path
        mean_and_std_dict = torch.load(statistics_path)
        mean = mean_and_std_dict.get('mean')
        std = mean_and_std_dict.get('std')

        
        if model_path.stem == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.fc = nn.Linear(2048, 1)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)

            test_data = get_test_data(config, mean, std, grayscale=False)
        elif model_path.stem == 'base_cnn':
            target_size = config['data_paths']['raw_data']['target_size']
            model = BaseCNN(target_size)
            model.load_state_dict(torch.load(model_path, weights_only=True))
            model.to(device)

            test_data = get_test_data(config, mean, std, grayscale=True)
        else:
            raise ValueError(f'Unknown model requested {model_path.stem}')
        
        model.eval()

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

        optimal_threshold = find_best_threshold(y_test, y_pred_proba, config['beta'])
        logger.info(f'Using optimal threshold={optimal_threshold}')
        y_pred = (y_pred_proba > optimal_threshold).astype('int32')
        class_labels = ['NORMAL', 'PNEUMONIA']
        report = classification_report(y_test, y_pred, target_names=[class_labels[0], class_labels[1]])
        logger.info(f'Classification report for {model_path.stem}')
        logger.info(f'\n{report}')

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

    results_df = pd.DataFrame(results).set_index('Model')
    logger.info(f'\n{results_df.to_string()}')

    plot_precision_recall_curve(recalls, precisions, labels, auprcs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    configs = parse_args_and_get_config('evaluate')
    evaluate(configs)
