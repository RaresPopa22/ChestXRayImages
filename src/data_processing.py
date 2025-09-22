import cv2
import numpy as np
import tensorflow as tf

from pathlib import Path

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_and_preprocess_split(base_path, split, target_size, grayscale_to_rgb):
    images = []
    labels = []
    label_map = {'NORMAL': 0, 'PNEUMONIA': 1}

    split_path = Path(base_path) / split

    for label_name, label_idx in label_map.items():
        label_path = split_path / label_name

        if not label_path.is_dir():
            print(f'Warning: Directory not found, skipping: {label_path}')
            continue

        for img_path in label_path.glob('*'):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.resize(img, (target_size, target_size))
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)

                if grayscale_to_rgb:
                    tensor_img = tf.convert_to_tensor(img)
                    img = tf.image.grayscale_to_rgb(tensor_img).numpy()

                images.append(img)
                labels.append(label_idx)

    return np.array(images), np.array(labels)


def perform_data_augmentation(config, X_train, y_train):
    metadata_config = config['metadata']
    minority_class_label = metadata_config['minority_class']
    majority_class_label = metadata_config['majority_class']

    minority_mask = y_train == minority_class_label
    X_train_minority = X_train[minority_mask]
    y_train_minority = y_train[minority_mask]

    majority_mask = y_train == majority_class_label
    X_train_majority = X_train[majority_mask]
    y_train_majority = y_train[majority_mask]

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    num_to_generate = len(X_train_majority) - len(X_train_minority)
    augmented_images = []
    augmented_labels = []

    if num_to_generate > 0:
        print(f'Generating {num_to_generate} new minority samples')
        minority_dataset = tf.data.Dataset.from_tensor_slices(X_train_minority)
        augmented_dataset = minority_dataset.repeat().map(
            lambda x: data_augmentation(x, training=True),
            num_parallel_calls=tf.data.AUTOTUNE
        ).take(num_to_generate)

        for img in augmented_dataset:
            augmented_images.append(img.numpy())
            augmented_labels.append(minority_class_label)

        augmented_images = np.array(augmented_images)
        augmented_labels = np.array(augmented_labels)

        X_train_balanced = np.concatenate((X_train_majority, X_train_minority, augmented_images), axis=0)
        y_train_balanced = np.concatenate((y_train_majority, y_train_minority, augmented_labels), axis=0)

        X_train_balanced, y_train_balanced = shuffle(X_train_balanced, y_train_balanced, random_state=1)

        print(f'After augmentation. New total training sample: {len(X_train_balanced)}')
        print(f'New Class 0 (Minority) count: {np.sum(y_train_balanced == 0)}')
        print(f'New Class 1 (Majority) count: {np.sum(y_train_balanced == 1)}')

        return X_train_balanced, y_train_balanced
    else:
        print(
            "Dataset is already balanced or the minority class is not the actual minority here. No augmentation will be performed")
        return X_train, y_train


def preprocess_training_data(config, grayscale_to_rgb=False):
    path, target_size = get_preprocess_data(config)

    X_train, y_train = load_and_preprocess_split(path, 'train', target_size, grayscale_to_rgb)
    X_train, y_train = shuffle(X_train, y_train, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=config['split']['val_percent'], random_state=1, stratify=y_train
    )

    X_train, y_train = perform_data_augmentation(config, X_train, y_train)

    return X_train, X_val, y_train, y_val


def get_preprocess_data(config):
    raw_config = config['data_paths']['raw_data']
    path = raw_config['path']
    target_size = raw_config['target_size']

    return path, target_size


def preprocess_test_data(config, grayscale_to_rgb=False):
    path, target_size = get_preprocess_data(config)
    X_test, y_test = load_and_preprocess_split(path, 'test', target_size, grayscale_to_rgb)

    return X_test, y_test


def create_data_blueprint(config):
    raw_config = config['data_paths']['raw_data']
    base_path = Path(raw_config['path']) / 'train'
    class_names = [d.name for d in base_path.iterdir() if d.is_dir()]

    filepaths = []
    labels = []

    for class_name in class_names:
        class_dir = base_path / class_name
        for file in class_dir.glob('*'):
            filepaths.append(str(file))
            labels.append(class_name)

    df = DataFrame({'filepath': filepaths, 'label': labels})
    return df


def get_data_and_split(config):
    df = create_data_blueprint(config)
    print("Blueprint created. Total images found:", len(df))
    print(df.head())
    train_df, valid_df = train_test_split(
        df, test_size=config['split']['val_percent'], stratify=df['label'], random_state=1)

    return train_df, valid_df


def get_training_generators(config, grayscale=False):
    train_df, valid_df = get_data_and_split(config)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest',
        validation_split=0.2)

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    target_size = config['data_paths']['raw_data']['target_size']
    batch_size = config['hyperparameters']['batch_size']

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        color_mode= 'grayscale' if grayscale else 'rgb',
        target_size=(target_size, target_size),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    val_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='filepath',
        y_col='label',
        color_mode='grayscale' if grayscale else 'rgb',
        target_size=(target_size, target_size),
        class_mode='binary',
        batch_size=batch_size,
        shuffle=False
    )

    return train_generator, val_generator


def get_test_generators(config, grayscale=False):
    raw_config = config['data_paths']['raw_data']
    test_path = Path(raw_config['path']) / 'test'

    datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = datagen.flow_from_directory(
        directory=test_path,
        color_mode='grayscale' if grayscale else 'rgb',
        class_mode='binary',
        shuffle=False
    )

    return test_generator
