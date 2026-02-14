"""モデル設定の定義。"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """モデルのダウンロード設定。"""

    name: str
    url: str
    dest: Path
    archive: bool = False
    inner_dir: str | None = None


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

kokoro_tts_config = ModelConfig(
    name="Kokoro TTS (INT8)",
    url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    dest=Path("kokoro-tts") / "kokoro-v1.0.int8.onnx",
)

kokoro_voices_config = ModelConfig(
    name="Kokoro TTS Voices",
    url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    dest=Path("kokoro-tts") / "voices-v1.0.bin",
)

DEFAULT_MODELS: list[ModelConfig] = [
    silero_vad_config,
    sense_voice_config,
    campplus_config,
    kokoro_tts_config,
    kokoro_voices_config,
]
