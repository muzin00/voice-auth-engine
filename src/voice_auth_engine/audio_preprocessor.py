"""音声ファイルの前処理。"""

from __future__ import annotations

import io
from pathlib import Path
from typing import NamedTuple

import av
import numpy as np
import numpy.typing as npt

from voice_auth_engine.audio_validator import validate_extension

TARGET_SAMPLE_RATE = 16000


class AudioPreprocessError(Exception):
    """音声前処理の基底例外。"""


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


AudioInput = bytes | str | Path
"""音声入力の型エイリアス。bytes / ファイルパス を受け付ける。"""


def load_audio(audio: AudioInput) -> AudioData:
    """AudioInput を 16kHz モノラル int16 の AudioData に変換する。

    Args:
        audio: 音声入力。bytes / str / Path を受け付ける。

    Returns:
        AudioData: 変換済みの音声データ。

    Raises:
        TypeError: 未対応の入力型の場合。
        FileNotFoundError: ファイルが存在しない場合。
        UnsupportedExtensionError: 非対応の拡張子の場合。
        AudioDecodeError: デコードに失敗した場合。
    """
    if isinstance(audio, bytes):
        return decode_audio(audio)
    if isinstance(audio, (str, Path)):
        path = Path(audio)
        if not path.exists():
            raise FileNotFoundError(f"音声ファイルが見つかりません: {path}")
        validate_extension(path)
        return decode_audio(path.read_bytes())
    raise TypeError(f"未対応の入力型です: {type(audio)}")


def decode_audio(data: bytes, *, format: str | None = None) -> AudioData:
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
