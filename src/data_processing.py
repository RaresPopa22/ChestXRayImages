from pathlib import Path

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from pandas import DataFrame
from sklearn.model_selection import train_test_split


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
        fill_mode='nearest')

    valid_datagen = ImageDataGenerator(rescale=1. / 255)

    target_size = config['data_paths']['raw_data']['target_size']
    batch_size = config['hyperparameters']['batch_size']

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        color_mode='grayscale' if grayscale else 'rgb',
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
