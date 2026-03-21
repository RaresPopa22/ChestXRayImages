import numpy as np

from src.util import deep_merge, find_best_threshold, read_config, read_configs


class TestUtils:

    def test_deep_merge_flat_key_override(self, sample_base_config):
        override_config = {'A': 2}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 2

    def test_deep_merge_addin_new_keys(self, sample_base_config):
        override_config = {'B': 2}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 1
        assert config.get('B') == 2

    def test_deep_merge_nested_merge(self):
        base_config = {'A': {'B': {'C': 2, 'D': 3}}, 'E': 3}
        override_config = {'A': {'B': {'D': 9}}}
        config = deep_merge(base_config, override_config)
        assert config == {'A': {'B': {'C': 2, 'D': 9}}, 'E': 3}

    def test_deep_merge_empty(self, sample_base_config):
        override_config = {}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 1

    def test_deep_merge_not_mutated(self, sample_base_config):
        override_config = {'B': 2}
        deep_merge(sample_base_config, override_config)
        assert sample_base_config == {'A': 1}

    def test_find_best_threshold_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([.05, .1, .2, .3, .33, .9])
        threshold = find_best_threshold(y_true, y_pred, beta=1)
        assert threshold == .3

    def test_find_best_threshold_recall_more_important(self):
        y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
        y_pred = np.array([.4, .1, .2, .3, .7, .9, .4, .4, .31, .29])
        threshold = find_best_threshold(y_true, y_pred, beta=2)
        assert threshold == .3

    def test_find_best_threshold_precision_more_important(self):
        y_true = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
        y_pred = np.array([.4, .1, .2, .3, .5, .9, .5, .7, .31, .29])
        threshold = find_best_threshold(y_true, y_pred, beta=.5)
        assert threshold == .7

    def test_read_config_happy_path(self, tmp_path):
        yaml_path = tmp_path / 'mock.yaml'
        yaml_path.write_text("A:\n  B: 2\n  C: 3\nD: 5")

        config = read_config(yaml_path)
        assert config == {'A': {'B': 2, 'C': 3}, 'D': 5}

    def test_read_configs_happy_path(self, tmp_path):
        base_yaml_path = tmp_path / 'base_yaml_path.yaml'
        base_yaml_path.write_text("A:\n  B: 2\n  C: 3\nD: 5")
        override_yaml_path = tmp_path / 'override_yaml_path.yaml'
        override_yaml_path.write_text("A:\n  B: 10\nD: 13")

        config = read_configs(base_yaml_path, override_yaml_path)
        assert config == {'A': {'B': 10, 'C': 3}, 'D': 13}

    
    
    