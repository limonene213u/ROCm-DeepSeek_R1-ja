import pytest
from pathlib import Path
from Python.dl_dataset import JapaneseDatasetDownloader


def test_download_failure_no_fallback(monkeypatch, tmp_path):
    def raise_error(*args, **kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr("Python.dl_dataset.load_dataset", raise_error)
    downloader = JapaneseDatasetDownloader(output_dir=tmp_path, use_fallback=False)
    with pytest.raises(RuntimeError):
        downloader.download_wikipedia_ja(5)


def test_download_failure_with_fallback(monkeypatch, tmp_path):
    def raise_error(*args, **kwargs):
        raise RuntimeError("fail")
    monkeypatch.setattr("Python.dl_dataset.load_dataset", raise_error)
    downloader = JapaneseDatasetDownloader(output_dir=tmp_path, use_fallback=True)
    path = downloader.download_wikipedia_ja(5)
    assert Path(path).exists()
