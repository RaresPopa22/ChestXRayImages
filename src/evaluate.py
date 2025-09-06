import tensorflow as tf

from src.data_processing import preprocess_test_data
from src.util import parse_args_and_get_config


def evaluate(config):
    X_test, y_test = preprocess_test_data(config)
    model = tf.keras.models.load_model(config['model_output_paths']['model'])
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test accuracy: {test_accuracy:.4f}')


if __name__ == '__main__':
    config = parse_args_and_get_config('evaluate')
    evaluate(config)
