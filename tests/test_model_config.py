"""Tests for ModelConfig."""

from pathlib import Path

import pytest

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


class TestGetModelsDir:
    def test_env_var_takes_priority(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        env_dir = tmp_path / "env-models"
        monkeypatch.setenv("VOICE_AUTH_ENGINE_MODELS_DIR", str(env_dir))
        assert ModelConfig.get_models_dir() == env_dir

    def test_legacy_dir_when_exists_and_non_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        monkeypatch.delenv("VOICE_AUTH_ENGINE_MODELS_DIR", raising=False)
        legacy = tmp_path / "models"
        legacy.mkdir()
        (legacy / "dummy.onnx").write_text("dummy")
        monkeypatch.setattr("voice_auth_engine.model_config.PROJECT_ROOT", tmp_path)
        assert ModelConfig.get_models_dir() == legacy

    def test_legacy_dir_skipped_when_empty(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.delenv("VOICE_AUTH_ENGINE_MODELS_DIR", raising=False)
        legacy = tmp_path / "models"
        legacy.mkdir()
        monkeypatch.setattr("voice_auth_engine.model_config.PROJECT_ROOT", tmp_path)
        result = ModelConfig.get_models_dir()
        assert result != legacy
        assert result.name == "models"

    def test_falls_back_to_cache_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.delenv("VOICE_AUTH_ENGINE_MODELS_DIR", raising=False)
        # PROJECT_ROOT/models does not exist
        monkeypatch.setattr("voice_auth_engine.model_config.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            "voice_auth_engine.model_config.platformdirs.user_cache_dir",
            lambda app: str(tmp_path / "cache" / app),
        )
        result = ModelConfig.get_models_dir()
        assert result == tmp_path / "cache" / "voice-auth-engine" / "models"
