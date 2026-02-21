"""SenseVoice による音声認識。"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import sherpa_onnx

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.model_config import sense_voice_config


class SpeechRecognizerError(Exception):
    """音声認識の基底例外。"""


class RecognizerModelLoadError(SpeechRecognizerError):
    """音声認識モデルの読み込み失敗。"""


class RecognitionError(SpeechRecognizerError):
    """音声認識処理の失敗。"""


class TranscriptionResult(NamedTuple):
    """音声認識結果。"""

    text: str  # 認識テキスト


def transcribe(
    audio: AudioData,
    *,
    model_dir: str | Path | None = None,
    language: str = "ja",
) -> TranscriptionResult:
    """音声データからテキストを認識する。

    Args:
        audio: 前処理済み音声データ（16kHz モノラル int16）。
        model_dir: SenseVoice モデルディレクトリのパス。
            None の場合は sense_voice_config.path を使用。
            ディレクトリ内に model.int8.onnx と tokens.txt が必要。
        language: 認識言語。デフォルトは "ja"（日本語）。

    Returns:
        TranscriptionResult: 認識されたテキスト。

    Raises:
        RecognizerModelLoadError: モデルの読み込みに失敗した場合。
        RecognitionError: 音声認識処理に失敗した場合。
    """
    if model_dir is None:
        model_dir = sense_voice_config.path
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise RecognizerModelLoadError(
            f"SenseVoice モデルディレクトリが見つかりません: {model_dir}"
        )

    model_file = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"

    if not model_file.exists():
        raise RecognizerModelLoadError(f"モデルファイルが見つかりません: {model_file}")
    if not tokens_file.exists():
        raise RecognizerModelLoadError(f"トークンファイルが見つかりません: {tokens_file}")

    if len(audio.samples) == 0:
        raise RecognitionError("空の音声データは認識できません")

    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_file),
            tokens=str(tokens_file),
            language=language,
        )
    except Exception as exc:
        raise RecognizerModelLoadError(f"SenseVoice モデルの読み込みに失敗しました: {exc}") from exc

    stream = recognizer.create_stream()
    stream.accept_waveform(audio.sample_rate, audio.samples_f32)
    recognizer.decode_stream(stream)

    return TranscriptionResult(text=stream.result.text.strip())
