import numpy as np
import tensorflow as tf
from keras.src.applications.resnet import ResNet50
from keras.src.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau


def build_cnn_model(config):
    raw_config = config['data_paths']['raw_data']
    target_size = raw_config['target_size']
    input_shape = (target_size, target_size, 1)

    input_img = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.ZeroPadding2D(padding=(3,3))(input_img)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)

    x_shortcut = x

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x_shortcut = tf.keras.layers.Conv2D(filters=512, kernel_size=1, padding='same', strides=1)(x_shortcut)
    x_shortcut = tf.keras.layers.BatchNormalization()(x_shortcut)

    x = tf.keras.layers.Add()([x, x_shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    f = tf.keras.layers.GlobalAveragePooling2D()(x)
    dense = tf.keras.layers.Dense(units=128, activation='relu')(f)
    dropout = tf.keras.layers.Dropout(0.2)(dense)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(dropout)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)

    return model


def build_resnet_50(config):
    raw_config = config['data_paths']['raw_data']
    target_size = raw_config['target_size']
    input_shape = (target_size, target_size, 3)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)

    return model


def get_early_stopping():
    return EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )


def get_model_checkpoint(config):
    return ModelCheckpoint(
        filepath=config['model_output_paths']['model'],
        save_best_only=True,
        monitor='val_loss'
    )


def get_class_weight(config, y_train):
    class_weights = compute_class_weight(
        config['hyperparameters']['class_weight'],
        classes=np.unique(y_train),
        y=y_train
    )

    return dict(enumerate(class_weights))


def get_lr_scheduler():
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        verbose=1,
        min_lr=1e-6
    )
