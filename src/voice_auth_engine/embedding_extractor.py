"""CAM++ (3D-Speaker) による話者埋め込みベクトル抽出。"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import sherpa_onnx

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.model_config import campplus_config


class EmbeddingExtractorError(Exception):
    """埋め込み抽出の基底例外。"""


class EmbeddingModelLoadError(EmbeddingExtractorError):
    """埋め込みモデルの読み込み失敗。"""


class EmbeddingExtractionError(EmbeddingExtractorError):
    """埋め込み抽出処理の失敗。"""


class Embedding(NamedTuple):
    """話者埋め込みベクトル。"""

    values: npt.NDArray[np.float32]


def extract_embedding(
    audio: AudioData,
    *,
    model_path: str | Path | None = None,
) -> Embedding:
    """音声データから話者埋め込みベクトルを抽出する。

    Args:
        audio: 前処理済み音声データ（16kHz モノラル int16）。
        model_path: CAM++ モデルファイルのパス。None の場合はデフォルトを使用。

    Returns:
        Embedding: 話者埋め込みベクトル。

    Raises:
        EmbeddingModelLoadError: モデルの読み込みに失敗した場合。
        EmbeddingExtractionError: 埋め込み抽出処理に失敗した場合。
    """
    if model_path is None:
        model_path = campplus_config.path
    model_path = Path(model_path)

    if not model_path.exists():
        raise EmbeddingModelLoadError(f"埋め込みモデルファイルが見つかりません: {model_path}")

    try:
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model_path))
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    except Exception as exc:
        raise EmbeddingModelLoadError(f"埋め込みモデルの読み込みに失敗しました: {exc}") from exc

    if len(audio.samples) == 0:
        raise EmbeddingExtractionError("空の音声データからは埋め込みを抽出できません")

    # 最低 100ms の音声が必要
    min_samples = int(audio.sample_rate * 0.1)
    if len(audio.samples) < min_samples:
        raise EmbeddingExtractionError("音声が短すぎて埋め込みを抽出できません")

    # int16 → float32 正規化
    samples_f32 = audio.samples.astype(np.float32) / 32768.0

    stream = extractor.create_stream()
    stream.accept_waveform(audio.sample_rate, samples_f32)
    stream.input_finished()

    if not extractor.is_ready(stream):
        raise EmbeddingExtractionError("音声が短すぎて埋め込みを抽出できません")

    values = np.array(extractor.compute(stream), dtype=np.float32)
    return Embedding(values=values)
