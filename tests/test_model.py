import torch

from src.model import BaseCNN


class TestModel:

    def test_output_shape(self, sample_config_base_cnn):
        input = torch.randn(32, 1, 256, 256)
        model = BaseCNN(sample_config_base_cnn)
        output = model(input)
        assert output.shape == torch.Size([32, 1])

    def test_different_batch_sizes(self, sample_config_base_cnn):
        model = BaseCNN(sample_config_base_cnn)
        test_batch_sizes = [1, 16, 32]

        for batch_size in test_batch_sizes:
            input = torch.randn(batch_size, 1, 256, 256)
            output = model(input)
            assert output.shape == torch.Size([batch_size, 1])

    def test_different_input_sizes(self, sample_config_base_cnn):
        model = BaseCNN(sample_config_base_cnn)
        test_input_sizes = [90, 128, 210]

        for input_size in test_input_sizes:
            input = torch.randn(32, 1, input_size, input_size)
            output = model(input)
            assert output.shape == torch.Size([32, 1])

    def test_output_is_raw_logits(self, sample_config_base_cnn):
        input = torch.randn(32, 1, 256, 256)
        model = BaseCNN(sample_config_base_cnn)
        output = model(input)
        assert output.min() < 0.0
            
    