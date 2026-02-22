"""音声データのバリデーション。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voice_auth_engine.audio_preprocessor import AudioData

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".mp3", ".ogg", ".webm", ".aac", ".flac", ".m4a"}
)

DEFAULT_MIN_SPEECH_SECONDS: float = 0.5


class AudioValidationError(Exception):
    """音声バリデーションの基底例外。"""


class EmptyAudioError(AudioValidationError):
    """音声サンプルが空。"""


class InsufficientDurationError(AudioValidationError):
    """発話時間が不足。"""

    def __init__(self, duration_seconds: float, min_seconds: float) -> None:
        self.duration_seconds = duration_seconds
        self.min_seconds = min_seconds
        super().__init__(f"発話時間が不足しています: {duration_seconds:.3f}s < {min_seconds}s")


class UnsupportedExtensionError(AudioValidationError):
    """非対応の拡張子。"""

    def __init__(self, extension: str) -> None:
        self.extension = extension
        super().__init__(f"非対応の拡張子です: {extension}")


def validate_audio(
    audio: AudioData,
    *,
    min_seconds: float = DEFAULT_MIN_SPEECH_SECONDS,
) -> None:
    """音声データの空チェックと発話時間の閾値チェックを行う。

    Args:
        audio: 音声データ。
        min_seconds: 必要な最小発話秒数。

    Raises:
        EmptyAudioError: サンプルが空の場合。
        InsufficientDurationError: 発話時間が不足の場合。
    """
    if len(audio.samples) == 0:
        raise EmptyAudioError("音声サンプルが空です")

    duration_seconds = len(audio.samples) / audio.sample_rate

    if duration_seconds < min_seconds:
        raise InsufficientDurationError(duration_seconds, min_seconds)


def validate_extension(path: str | Path) -> str:
    """ファイル拡張子がサポート対象か検証する。

    Args:
        path: ファイルパス。

    Returns:
        正規化された小文字の拡張子（例: ".wav"）。

    Raises:
        UnsupportedExtensionError: 非対応の拡張子の場合。
    """
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedExtensionError(ext)
    return ext
