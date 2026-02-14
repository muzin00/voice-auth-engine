"""Tests for ModelDownloader."""

from pathlib import Path

from voice_auth_engine.model_config import DEFAULT_MODELS, ModelConfig
from voice_auth_engine.model_downloader import ModelDownloader


class TestModelDownloader:
    def test_uses_default_models_when_none_provided(self, tmp_path: Path):
        downloader = ModelDownloader(models_dir=tmp_path)
        assert downloader.models is DEFAULT_MODELS

    def test_uses_custom_models_when_provided(self, tmp_path: Path):
        custom = [
            ModelConfig(
                name="custom",
                url="https://example.com/m.onnx",
                dest=Path("m.onnx"),
            )
        ]
        downloader = ModelDownloader(models_dir=tmp_path, models=custom)
        assert downloader.models is custom

    def test_is_downloaded_file_exists(self, tmp_path: Path):
        model = ModelConfig(
            name="test",
            url="https://example.com/model.onnx",
            dest=Path("sub") / "model.onnx",
        )
        dest = tmp_path / model.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("dummy")

        downloader = ModelDownloader(models_dir=tmp_path)
        assert downloader.is_downloaded(model) is True

    def test_is_downloaded_file_missing(self, tmp_path: Path):
        model = ModelConfig(
            name="test",
            url="https://example.com/model.onnx",
            dest=Path("sub") / "model.onnx",
        )
        downloader = ModelDownloader(models_dir=tmp_path)
        assert downloader.is_downloaded(model) is False

    def test_is_downloaded_archive_with_files(self, tmp_path: Path):
        model = ModelConfig(
            name="test",
            url="https://example.com/model.tar.bz2",
            dest=Path("archive-dir"),
            archive=True,
            inner_dir="inner",
        )
        dest = tmp_path / model.dest
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "some_file.txt").write_text("dummy")

        downloader = ModelDownloader(models_dir=tmp_path)
        assert downloader.is_downloaded(model) is True

    def test_is_downloaded_archive_empty_dir(self, tmp_path: Path):
        model = ModelConfig(
            name="test",
            url="https://example.com/model.tar.bz2",
            dest=Path("archive-dir"),
            archive=True,
            inner_dir="inner",
        )
        dest = tmp_path / model.dest
        dest.mkdir(parents=True, exist_ok=True)

        downloader = ModelDownloader(models_dir=tmp_path)
        assert downloader.is_downloaded(model) is False
