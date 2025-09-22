import numpy as np
from sklearn.utils import compute_class_weight
from tensorflow.keras.optimizers import Adam

from src.data_processing import get_training_generators
from src.model import build_cnn_model, get_early_stopping, get_model_checkpoint, build_resnet_50, get_lr_scheduler
from src.util import parse_args_and_get_config


def train(config):
    model_name = config['model_name']

    if model_name == 'cnn':
        train_generator, val_generator = get_training_generators(config, grayscale=True)
        model = build_cnn_model(config)
    elif model_name == 'resnet50':
        train_generator, val_generator = get_training_generators(config)
        model = build_resnet_50(config)

    class_labels = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

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
        class_weight=class_weight_dict if model_name == 'resnet50' else None,
        callbacks=[early_stopping, model_checkpoint, lr_scheduler]
    )


if __name__ == '__main__':
    config = parse_args_and_get_config('train')
    train(config)
