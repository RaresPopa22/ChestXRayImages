from src.data_processing import preprocess_training_data, create_datagen, create_data_generators, get_generators
from src.model import build_cnn_model, get_early_stopping, get_model_checkpoint, build_resnet_50, get_lr_scheduler
from src.util import parse_args_and_get_config
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train(config):
    model_name = config['model_name']

    if model_name == 'cnn':
        X_train, X_val, y_train, y_val = preprocess_training_data(config)
        model = build_cnn_model(config)
        optimizer = Adam(learning_rate=1e-5)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

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
    elif model_name == 'resnet50':
        train_generator, val_generator = get_generators(config)

        model = build_resnet_50(config)

        optimizer = Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = get_early_stopping()
        model_checkpoint = get_model_checkpoint(config)
        lr_scheduler = get_lr_scheduler()

        hyperparameters = config['hyperparameters']
        history = model.fit(
            train_generator,
            epochs=hyperparameters['epochs'],
            validation_data=val_generator,
            callbacks=[early_stopping, model_checkpoint, lr_scheduler]
        )


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train(config)
