import logging
import time

import torch
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, Resize, RandomRotation, RandomAffine, RandomResizedCrop, ToTensor, Grayscale, Normalize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset


class DatasetWrapper(Dataset):
    def __init__(self, subset, transform):
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        image, label = self.subset[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    

def get_data_and_idxs(config):
    data = ImageFolder(config['data_paths']['raw_data']['training_data_path'])
    indices = list(range(len(data)))
    split_percentage = config['split']['val_percent']
    train_idxs, val_idxs = train_test_split(indices, stratify=data.targets, test_size=split_percentage, random_state=1)

    return data, train_idxs, val_idxs


def compute_mean_and_std(data, train_idxs, config, grayscale=True):
    logger = logging.getLogger(__name__)
    logger.info('Computing mean and standard deviation on the training set...')
    start = time.time()

    target_size = config['data_paths']['raw_data']['target_size']
    batch_size = config['hyperparameters']['batch_size']
    num_workers = config['num_workers']

    deterministic_transform_list = [
        Resize((target_size, target_size)),
        ToTensor()
    ]

    if grayscale:
        deterministic_transform_list.insert(0, Grayscale())
        sum_pixels = torch.tensor(0.0)
        sum_squares = torch.tensor(0.0)
    else:
        sum_pixels = torch.zeros(3)
        sum_squares = torch.zeros(3)
    
    deterministic_transform = Compose(deterministic_transform_list)
    training_dataset = DatasetWrapper(Subset(data, train_idxs), deterministic_transform)
    training_data_loader = DataLoader(training_dataset, batch_size, shuffle=False, num_workers=num_workers)

    N = 0.0
    for batch_X, _ in training_data_loader:
        if grayscale:
            sum_pixels += batch_X.sum()
            sum_squares += (batch_X ** 2).sum()
            N += batch_X.nelement()
        else:
            sum_pixels += batch_X.sum(dim=[0, 2, 3])
            sum_squares += (batch_X ** 2).sum(dim=[0, 2, 3])
            N += batch_X.shape[0] * batch_X.shape[2] * batch_X.shape[3]

    mean = sum_pixels / N
    std_squared = (1 / N) * sum_squares - mean ** 2
    std = torch.sqrt(std_squared)
    

    logger.info(f'Done: mean={mean} and std={std} in {time.time() - start} seconds')
    path = config['data_paths']['processed']['mean_and_std_gray'] if grayscale else config['data_paths']['processed']['mean_and_std_color']
    torch.save({'mean': mean, 'std': std}, path)

    return mean, std


def get_train_val_data(config, data, train_idxs, val_idxs, mean, std, grayscale=True):
    target_size = config['data_paths']['raw_data']['target_size']
    batch_size = config['hyperparameters']['batch_size']

    training_transform_list = [
        RandomAffine(10, translate=(0.1, 0.1), shear=5.7),
        RandomResizedCrop(size=target_size ,scale=(0.9, 1)),
        ToTensor(),
        Normalize(mean, std)
    ]

    eval_transform_list = [
        Resize((target_size, target_size)),
        ToTensor(),
        Normalize(mean, std)
    ]

    if grayscale:
        training_transform_list.insert(0, Grayscale())
        eval_transform_list.insert(0, Grayscale())

    training_transform = Compose(training_transform_list)
    eval_transform = Compose(eval_transform_list)

    training_wrapper = DatasetWrapper(Subset(data, train_idxs), training_transform)
    eval_wrapper = DatasetWrapper(Subset(data, val_idxs), eval_transform)
    
    training_set = DataLoader(training_wrapper, batch_size, shuffle=True, num_workers=config['num_workers'], persistent_workers=True)
    eval_set = DataLoader(eval_wrapper, batch_size, shuffle=False, num_workers=config['num_workers'], persistent_workers=True)

    return training_set, eval_set


def get_test_data(config, mean, std, grayscale=True):
    target_size = config['data_paths']['raw_data']['target_size']
    batch_size = config['hyperparameters']['batch_size']

    test_transform_list = [
        Resize((target_size, target_size)),
        ToTensor(),
        Normalize(mean, std)
    ]

    if grayscale:
        test_transform_list.insert(0, Grayscale())

    test_transform = Compose(test_transform_list)
    data = ImageFolder(config['data_paths']['raw_data']['test_data_path'], transform=test_transform)

    return DataLoader(
        data, 
        batch_size, 
        shuffle=False,  
        num_workers=config['num_workers'], 
        persistent_workers=True
    )