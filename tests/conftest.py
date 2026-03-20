import pytest
from PIL import Image


@pytest.fixture
def sample_base_config():
    return {'A': 1}


@pytest.fixture
def sample_mean_and_std_paths(tmp_path):
    gray_path = tmp_path / 'mean_and_std_gray.pt'
    color_path = tmp_path / 'mean_and_std_color.pt'

    return gray_path, color_path
    

@pytest.fixture
def sample_processing_config(sample_mean_and_std_paths):
    gray_path, color_path = sample_mean_and_std_paths
    return {
            'data_paths': {'raw_data' : {'target_size': 256}, 'processed':
                           {'mean_and_std_gray': gray_path, 'mean_and_std_color': color_path}},
            'hyperparameters': {'batch_size': 2},
            'data_processing': {'num_workers': 1}
        }


@pytest.fixture
def sample_processing_data_gray():
    return [
            (Image.new('RGB', (256, 256), color=(0, 0, 0)), '1'),
            (Image.new('RGB', (256, 256), color=(0, 0, 0)), '1'),
            (Image.new('RGB', (256, 256), color=(0, 0, 0)), '1'),
            (Image.new('RGB', (256, 256), color=(255, 255, 255)), '0'),
            (Image.new('RGB', (256, 256), color=(255, 255, 255)), '0'),
            (Image.new('RGB', (256, 256), color=(255, 255, 255)), '0'),
        ]


@pytest.fixture
def sample_processing_data_color():
    return [
            (Image.new('RGB', (256, 256), color=(153, 128, 102)), '1'),
            (Image.new('RGB', (256, 256), color=(128, 77, 153)), '1'),
            (Image.new('RGB', (256, 256), color=(102, 51, 26)), '1'),
            (Image.new('RGB', (256, 256), color=(128, 77, 153)), '0'),
            (Image.new('RGB', (256, 256), color=(51, 153, 255)), '0'),
            (Image.new('RGB', (256, 256), color=(153, 128, 255)), '0'),
        ]


class MockResponse:
    def __init__(self, targets):
        self.targets = targets


@pytest.fixture
def sample_targets():
    return MockResponse(targets=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ,0 ,0])


@pytest.fixture
def sample_dataset():
    return [(1, 'a'), (2, 'b'), (3, 'c')]


@pytest.fixture
def sample_transform():
    return lambda x: x * 2