"""テスト用音声データ・ファイル生成ヘルパー。"""

from __future__ import annotations

from pathlib import Path

import av
import av.audio
import numpy as np
import numpy.typing as npt

from voice_auth_engine.audio_preprocessor import AudioData

TARGET_SAMPLE_RATE = 16000


def generate_voiced_samples(
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    duration: float = 1.0,
    f0: float = 150.0,
) -> npt.NDArray[np.int16]:
    """VAD が反応する発話風の複合信号を生成する。

    基本周波数 + 倍音の複合信号。単純なサイン波では VAD が反応しない。
    """
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)
    signal = np.zeros(n, dtype=np.float32)
    for h in range(1, 6):
        signal += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t)
    signal /= np.max(np.abs(signal))
    return (signal * 30000).astype(np.int16)


def generate_silence_samples(
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    duration: float = 1.0,
) -> npt.NDArray[np.int16]:
    """無音信号を生成する。"""
    return np.zeros(int(sample_rate * duration), dtype=np.int16)


def make_audio_data(
    samples: npt.NDArray[np.int16],
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> AudioData:
    """AudioData を生成する。"""
    return AudioData(samples=samples, sample_rate=sample_rate)


def generate_audio_file(
    path: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    duration: float = 0.5,
    codec: str = "pcm_s16le",
    format_name: str = "wav",
) -> Path:
    """PyAV を使ってテスト用音声ファイルを生成する。

    Args:
        path: 出力ファイルパス。
        sample_rate: サンプルレート。
        channels: チャンネル数。
        duration: 秒数。
        codec: 音声コーデック名。
        format_name: コンテナフォーマット名。

    Returns:
        生成されたファイルのパス。
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
    # 440Hz サイン波
    mono = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    if channels == 2:
        samples = np.stack([mono, mono])
        fmt = "s16p"
    else:
        samples = mono.reshape(1, -1)
        fmt = "s16"

    layout = "stereo" if channels == 2 else "mono"

    with av.open(str(path), mode="w", format=format_name) as container:
        stream = container.add_stream(codec, rate=sample_rate)
        assert isinstance(stream, av.audio.AudioStream)
        stream.layout = layout

        frame = av.AudioFrame.from_ndarray(samples, format=fmt, layout=layout)
        frame.sample_rate = sample_rate

        for packet in stream.encode(frame):
            container.mux(packet)

        for packet in stream.encode(None):
            container.mux(packet)

    return path
