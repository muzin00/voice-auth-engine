"""embedding_extractor モジュールのテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.embedding_extractor import (
    Embedding,
    EmbeddingExtractionError,
    EmbeddingModelLoadError,
    extract_embedding,
)

from .conftest import requires_campplus_model


class TestExtractEmbedding:
    """extract_embedding 関数のテスト。"""

    def test_model_path_not_found(self, voiced_audio: AudioData) -> None:
        """不正パスで EmbeddingModelLoadError が発生する。"""
        with pytest.raises(EmbeddingModelLoadError, match="モデルファイルが見つかりません"):
            extract_embedding(voiced_audio, model_path="/nonexistent/path")

    @requires_campplus_model
    def test_output_dimension(self, voiced_audio_3s: AudioData) -> None:
        """出力ベクトルが192次元である。"""
        result = extract_embedding(voiced_audio_3s)
        assert result.values.shape == (192,)

    @requires_campplus_model
    def test_output_dtype_is_float32(self, voiced_audio_3s: AudioData) -> None:
        """出力の dtype が float32 である。"""
        result = extract_embedding(voiced_audio_3s)
        assert result.values.dtype == np.float32

    @requires_campplus_model
    def test_same_audio_produces_same_embedding(self, voiced_audio_3s: AudioData) -> None:
        """同一音声から同一の埋め込みベクトルが得られる。"""
        result1 = extract_embedding(voiced_audio_3s)
        result2 = extract_embedding(voiced_audio_3s)
        np.testing.assert_array_equal(result1.values, result2.values)

    @requires_campplus_model
    def test_result_type(self, voiced_audio_3s: AudioData) -> None:
        """戻り値が Embedding である。"""
        result = extract_embedding(voiced_audio_3s)
        assert isinstance(result, Embedding)

    @requires_campplus_model
    def test_empty_audio_raises_error(self) -> None:
        """空音声で EmbeddingExtractionError が発生する。"""
        empty = AudioData(samples=np.array([], dtype=np.int16), sample_rate=16000)
        with pytest.raises(EmbeddingExtractionError, match="空の音声データ"):
            extract_embedding(empty)

    @requires_campplus_model
    def test_short_audio_raises_error(self) -> None:
        """短すぎる音声で EmbeddingExtractionError が発生する。"""
        # 50ms の極短音声 (100ms 未満)
        short = AudioData(
            samples=np.ones(800, dtype=np.int16),
            sample_rate=16000,
        )
        with pytest.raises(EmbeddingExtractionError, match="短すぎて"):
            extract_embedding(short)
