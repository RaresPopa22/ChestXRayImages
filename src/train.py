from src.data_processing import preprocess_training_data
from src.model import build_cnn_model, get_early_stopping, get_model_checkpoint
from src.util import parse_args_and_get_config


def train(config):
    X_train, X_val, y_train, y_val = preprocess_training_data(config)

    model = build_cnn_model(config)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = get_early_stopping()
    model_checkpoint = get_model_checkpoint(config)

    hyperparameters = config['hyperparameters']
    history = model.fit(
        X_train, y_train,
        epochs=hyperparameters['epochs'],
        validation_data=(X_val, y_val),
        batch_size=hyperparameters['batch_size'],
        callbacks=[early_stopping, model_checkpoint]
    )


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train(config)
