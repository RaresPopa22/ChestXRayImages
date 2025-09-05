import tensorflow as tf
from src.util import read_config


def build_cnn_model(config):
    raw_config = config['data_paths']['raw_data']
    target_size = raw_config['target_size']
    input_shape = (target_size, target_size, 1)

    input_img = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(input_img)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    f = tf.keras.layers.Flatten()(x)
    dense = tf.keras.layers.Dense(units=128, activation='relu')(f)
    dropout = tf.keras.layers.Dropout(0.5)(dense)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(dropout)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)

    return model


if __name__ == '__main__':
    config = read_config("../config/base_config.yaml")
    model = build_cnn_model(config)
    model.summary()
