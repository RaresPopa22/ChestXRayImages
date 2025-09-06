from src.data_processing import preprocess_training_data
from src.model import build_cnn_model
from src.util import parse_args_and_get_config


def train(config):
    X_train, X_val, y_train, y_val = preprocess_training_data(config)

    model = build_cnn_model(config)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

    model.save(config['model_output_paths']['model'])


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train(config)
