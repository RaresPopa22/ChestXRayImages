import torch

from src.data_processing import DatasetWrapper, compute_mean_and_std, get_imbalance_ratio


class TestDataProcessing:

    def test_compute_mean_and_std_gray(self, sample_processing_config, sample_processing_data_gray, sample_mean_and_std_paths):
        mean, std = compute_mean_and_std(
                sample_processing_data_gray, range(0, len(sample_processing_data_gray)), sample_processing_config, True
            )
        
        assert torch.allclose(mean, torch.tensor(0.5), atol=1e-4)
        assert torch.allclose(std, torch.tensor(0.5), atol=1e-4)

        gray_path, _ = sample_mean_and_std_paths

        mean_and_std_dict = torch.load(str(gray_path)) 
        assert torch.allclose(mean_and_std_dict.get('mean'), torch.tensor(.5), atol=1e-4)
        assert torch.allclose(mean_and_std_dict.get('std'), torch.tensor(.5), atol=1e-4)



    def test_compute_mean_and_std_color(self, sample_processing_config, sample_processing_data_color, sample_mean_and_std_paths):
        mean, std = compute_mean_and_std(
                sample_processing_data_color, range(0, len(sample_processing_data_color)), sample_processing_config, False
            )
        
        assert torch.allclose(mean, torch.tensor([.4673, .4013, .617]), atol=1e-4)
        assert torch.allclose(std, torch.tensor([.1376, .1414, .3179]), atol=1e-4)

        _, color_path = sample_mean_and_std_paths

        mean_and_std_dict = torch.load(str(color_path)) 
        assert torch.allclose(mean_and_std_dict.get('mean'), torch.tensor([.4673, .4013, .617]), atol=1e-4)
        assert torch.allclose(mean_and_std_dict.get('std'), torch.tensor([.1376, .1414, .3179]), atol=1e-4)


    def test_get_imbalance_ratio(self, sample_targets):
        idxs = range(len(sample_targets.targets))
        ratio = get_imbalance_ratio(sample_targets, idxs)
        assert ratio == .2


    def test_dataset_wrapper_len(self, sample_dataset, sample_transform):
        wrapper = DatasetWrapper(sample_dataset, sample_transform)
        assert len(wrapper) == 3


    def test_dataset_wrapper_getitem(self, sample_dataset, sample_transform):
        wrapper= DatasetWrapper(sample_dataset, sample_transform)
        
        for i in range(len(wrapper)):
            image, label = wrapper[i]
            assert image == sample_dataset[i][0] * 2
            assert label == sample_dataset[i][1] 
        