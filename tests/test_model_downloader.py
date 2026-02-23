"""Tests for ModelDownloader."""

from pathlib import Path
from unittest.mock import patch

import pytest

from voice_auth_engine.model_config import DEFAULT_MODELS, ModelConfig
from voice_auth_engine.model_downloader import ModelDownloader, ModelDownloadError


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


class TestEnsureDownload:
    def test_returns_path_when_already_downloaded(self, tmp_path: Path):
        model = ModelConfig(
            name="test",
            url="https://example.com/model.onnx",
            dest=Path("sub") / "model.onnx",
        )
        dest = tmp_path / model.dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("dummy")

        downloader = ModelDownloader(models_dir=tmp_path)
        result = downloader.ensure_download(model)
        assert result == dest

    def test_downloads_when_not_present(self, tmp_path: Path):
        model = ModelConfig(
            name="test",
            url="https://example.com/model.onnx",
            dest=Path("sub") / "model.onnx",
        )

        def fake_download(self, m):
            dest = self.models_dir / m.dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("downloaded")

        with patch.object(ModelDownloader, "download", fake_download):
            downloader = ModelDownloader(models_dir=tmp_path)
            result = downloader.ensure_download(model)

        assert result == tmp_path / model.dest

    def test_raises_model_download_error_on_failure(self, tmp_path: Path):
        model = ModelConfig(
            name="test",
            url="https://example.com/model.onnx",
            dest=Path("sub") / "model.onnx",
        )

        with (
            patch.object(ModelDownloader, "download", side_effect=RuntimeError("network error")),
            pytest.raises(ModelDownloadError, match="network error"),
        ):
            downloader = ModelDownloader(models_dir=tmp_path)
            downloader.ensure_download(model)


class TestModelsDir:
    def test_defaults_to_get_models_dir(self, tmp_path: Path):
        with patch.object(ModelConfig, "get_models_dir", return_value=tmp_path):
            downloader = ModelDownloader()
        assert downloader.models_dir == tmp_path
