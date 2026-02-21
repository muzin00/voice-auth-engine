"""Silero VAD による音声区間検出。"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import sherpa_onnx

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.model_config import silero_vad_config


class VadError(Exception):
    """VAD 処理の基底例外。"""


class VadModelLoadError(VadError):
    """VAD モデルの読み込み失敗。"""


class SpeechSegment(NamedTuple):
    """検出された発話区間。"""

    start_sample: int  # 開始サンプルインデックス
    end_sample: int  # 終了サンプルインデックス (exclusive)
    start_sec: float  # 開始時刻 (秒)
    end_sec: float  # 終了時刻 (秒)


class SpeechSegments(NamedTuple):
    """発話区間検出の結果。"""

    segments: list[SpeechSegment]
    audio: AudioData  # 元の音声データへの参照


def detect_speech(
    audio: AudioData,
    *,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
    min_speech_duration: float = 0.25,
    min_silence_duration: float = 0.5,
) -> SpeechSegments:
    """音声データから発話区間を検出する。

    Args:
        audio: 前処理済み音声データ（16kHz モノラル int16）。
        model_path: Silero VAD モデルファイルのパス。None の場合はデフォルトを使用。
        threshold: 発話検出の閾値（0.0〜1.0）。
        min_speech_duration: 最小発話持続時間（秒）。
        min_silence_duration: 最小無音持続時間（秒）。

    Returns:
        SpeechSegments: 検出された発話区間のリストと元の音声データ。

    Raises:
        VadModelLoadError: モデルの読み込みに失敗した場合。
    """
    if model_path is None:
        model_path = silero_vad_config.path
    model_path = Path(model_path)

    if not model_path.exists():
        raise VadModelLoadError(f"VAD モデルファイルが見つかりません: {model_path}")

    config = sherpa_onnx.VadModelConfig(
        silero_vad=sherpa_onnx.SileroVadModelConfig(
            model=str(model_path),
            threshold=threshold,
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
        ),
        sample_rate=audio.sample_rate,
    )

    try:
        vad = sherpa_onnx.VoiceActivityDetector(config)
    except Exception as exc:
        raise VadModelLoadError(f"VAD モデルの読み込みに失敗しました: {exc}") from exc

    if len(audio.samples) == 0:
        return SpeechSegments(segments=[], audio=audio)

    vad.accept_waveform(audio.samples_f32)
    vad.flush()

    segments: list[SpeechSegment] = []
    while not vad.empty():
        seg = vad.front
        start_sample = seg.start
        end_sample = start_sample + len(seg.samples)
        # end_sample が音声長を超えないようにクランプ
        end_sample = min(end_sample, len(audio.samples))
        segments.append(
            SpeechSegment(
                start_sample=start_sample,
                end_sample=end_sample,
                start_sec=start_sample / audio.sample_rate,
                end_sec=end_sample / audio.sample_rate,
            )
        )
        vad.pop()

    return SpeechSegments(segments=segments, audio=audio)


def extract_speech(segments: SpeechSegments) -> AudioData:
    """検出された発話区間の音声を切り出して結合する。

    Args:
        segments: detect_speech の結果。

    Returns:
        AudioData: 発話区間のみを結合した音声データ。
    """
    if not segments.segments:
        return AudioData(
            samples=np.array([], dtype=np.int16),
            sample_rate=segments.audio.sample_rate,
        )

    parts = [segments.audio.samples[seg.start_sample : seg.end_sample] for seg in segments.segments]
    return AudioData(
        samples=np.concatenate(parts),
        sample_rate=segments.audio.sample_rate,
    )
