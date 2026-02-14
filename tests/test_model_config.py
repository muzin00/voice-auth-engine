"""Tests for ModelConfig and DEFAULT_MODELS."""

from pathlib import Path

from voice_auth_engine.model_config import ModelConfig


class TestModelConfig:
    def test_create_simple_model(self):
        config = ModelConfig(
            name="test-model",
            url="https://example.com/model.onnx",
            dest=Path("test") / "model.onnx",
        )
        assert config.name == "test-model"
        assert config.url == "https://example.com/model.onnx"
        assert config.dest == Path("test") / "model.onnx"
        assert config.archive is False
        assert config.inner_dir is None

    def test_create_archive_model(self):
        config = ModelConfig(
            name="archive-model",
            url="https://example.com/model.tar.bz2",
            dest=Path("archive"),
            archive=True,
            inner_dir="inner",
        )
        assert config.archive is True
        assert config.inner_dir == "inner"
