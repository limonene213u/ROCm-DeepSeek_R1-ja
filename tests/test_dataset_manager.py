import json
from pathlib import Path

from Python.deepseek_ja_adapter import DatasetManager, JapaneseDataConfig, ExecutionMode


def test_sample_dataset_creation(tmp_path):
    cfg = JapaneseDataConfig(base_dir=tmp_path, execution_mode=ExecutionMode.DEVELOPMENT)
    manager = DatasetManager(cfg)
    assert manager.ensure_datasets_exist() is True
    for fname in cfg.train_files:
        assert (tmp_path / fname).exists()
    assert (tmp_path / 'persona_config.json').exists()


def test_production_missing_files(tmp_path):
    cfg = JapaneseDataConfig(base_dir=tmp_path, execution_mode=ExecutionMode.PRODUCTION)
    manager = DatasetManager(cfg)
    assert manager.ensure_datasets_exist() is False
    for fname in cfg.train_files:
        assert not (tmp_path / fname).exists()
