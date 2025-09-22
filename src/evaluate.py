import pandas as pd

from pathlib import Path
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, roc_auc_score, \
    f1_score

from src.data_processing import preprocess_test_data, get_test_generators
from src.util import parse_args_and_get_config, plot_precision_recall_curve, find_best_threshold
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def evaluate(config, model_paths):
    X_test, y_test = preprocess_test_data(config)
    test_generator = X_test

    results, precisions, recalls, labels, auprcs = [], [], [], [], []

    for path in model_paths:
        model_path = Path(path)
        model = tf.keras.models.load_model(model_path)

        if model_path.stem == 'resnet50':
            test_generator = get_test_generators(config)
        elif model_path.stem == 'cnn':
            test_generator = get_test_generators(config, grayscale=True)

        y_test = test_generator.classes

        y_pred_proba = model.predict(test_generator)

        optimal_threshold = find_best_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba > optimal_threshold).astype('int32')
        class_labels = ['NORMAL', 'PNEUMONIA']
        report = classification_report(y_test, y_pred, target_names=[class_labels[0], class_labels[1]])
        print(f'Printing the report for {model_path.stem}')
        print(report)

        auprc = average_precision_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        results.append({
            'Model': model_path.stem,
            'AUPRC': auprc,
            'ROC AUC': roc_auc,
            'F1-Score (PNEUMONIA)': f1
        })

        precision, recall, _ = precision_recall_curve(y_test, y_pred)
        recalls.append(recall)
        precisions.append(precision)
        labels.append(model_path.stem)
        auprcs.append(auprc)

    results_df = pd.DataFrame(results).set_index('Model')
    pd.set_option('display.max_columns', None)
    print(results_df.to_string(index=False))

    plot_precision_recall_curve(recalls, precisions, labels, auprcs)


if __name__ == '__main__':
    config, model_paths = parse_args_and_get_config('evaluate')
    evaluate(config, model_paths)
