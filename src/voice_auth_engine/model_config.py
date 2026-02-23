"""モデル設定の定義。"""

import os
from dataclasses import dataclass
from pathlib import Path

import platformdirs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class ModelConfig:
    """モデルのダウンロード設定。"""

    name: str
    url: str
    dest: Path
    archive: bool = False
    inner_dir: str | None = None

    @property
    def path(self) -> Path:
        """モデルファイルの絶対パスを返す。"""
        return self.get_models_dir() / self.dest

    @staticmethod
    def get_models_dir() -> Path:
        """モデルディレクトリを解決する。

        優先順位:
        1. 環境変数 VOICE_AUTH_ENGINE_MODELS_DIR
        2. リポジトリルート models/（存在 & 中身がある場合のみ、開発用後方互換）
        3. OS 標準キャッシュ: platformdirs.user_cache_dir("voice-auth-engine") / "models"
        """
        env = os.environ.get("VOICE_AUTH_ENGINE_MODELS_DIR")
        if env:
            return Path(env)

        legacy = PROJECT_ROOT / "models"
        if legacy.is_dir() and any(legacy.iterdir()):
            return legacy

        return Path(platformdirs.user_cache_dir("voice-auth-engine")) / "models"


silero_vad_config = ModelConfig(
    name="Silero VAD",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    dest=Path("silero-vad") / "silero_vad.onnx",
)

sense_voice_config = ModelConfig(
    name="SenseVoice",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2",
    dest=Path("sense-voice"),
    archive=True,
    inner_dir="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
)

campplus_config = ModelConfig(
    name="CAM++ (3D-Speaker)",
    url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
    dest=Path("3dspeaker") / "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
)

DEFAULT_MODELS: list[ModelConfig] = [
    silero_vad_config,
    sense_voice_config,
    campplus_config,
]
