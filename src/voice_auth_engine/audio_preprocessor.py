"""音声ファイルの前処理。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np
import numpy.typing as npt

TARGET_SAMPLE_RATE = 16000

SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".webm", ".aac", ".flac", ".m4a"})


class AudioPreprocessError(Exception):
    """音声前処理の基底例外。"""


class UnsupportedFormatError(AudioPreprocessError):
    """非対応の音声フォーマット。"""


class AudioDecodeError(AudioPreprocessError):
    """音声デコード失敗。"""


class AudioData(NamedTuple):
    """前処理済み音声データ。"""

    samples: npt.NDArray[np.int16]  # 16kHz モノラル int16
    sample_rate: int  # 常に 16000

    @property
    def samples_f32(self) -> npt.NDArray[np.float32]:
        """int16 サンプルを [-1.0, 1.0] の float32 に正規化して返す。"""
        return self.samples.astype(np.float32) / 32768.0


def load_audio(path: str | Path) -> AudioData:
    """音声ファイルを読み込み、16kHz モノラル int16 に変換する。

    Args:
        path: 音声ファイルのパス。

    Returns:
        AudioData: 変換済みの音声データ。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        UnsupportedFormatError: 非対応の拡張子の場合。
        AudioDecodeError: デコードに失敗した場合。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"非対応の音声フォーマットです: {ext}")

    return load_audio_bytes(path.read_bytes())


def load_audio_bytes(data: bytes, *, format: str | None = None) -> AudioData:
    """bytes から音声をデコードし、16kHz モノラル int16 に変換する。

    Args:
        data: 音声データのバイト列。
        format: コンテナフォーマット名（例: "wav", "mp3"）。
            None の場合は PyAV が自動判別する。

    Returns:
        AudioData: 変換済みの音声データ。

    Raises:
        AudioDecodeError: デコードに失敗した場合。
    """
    if not data:
        raise AudioDecodeError("音声データが空です")

    try:
        with av.open(io.BytesIO(data), mode="r", format=format) as container:
            audio_stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SAMPLE_RATE)

            frames: list[npt.NDArray[np.int16]] = []

            for frame in container.decode(audio_stream):
                resampled = resampler.resample(frame)
                for r in resampled:
                    array = r.to_ndarray().flatten().astype(np.int16)
                    frames.append(array)

            # flush 残余サンプル
            for r in resampler.resample(None):
                array = r.to_ndarray().flatten().astype(np.int16)
                frames.append(array)

    except Exception as exc:
        raise AudioDecodeError(f"音声データのデコードに失敗しました: {exc}") from exc

    if not frames:
        raise AudioDecodeError("音声データが空です")

    samples = np.concatenate(frames)
    return AudioData(samples=samples, sample_rate=TARGET_SAMPLE_RATE)
