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

from .audio_factory import make_embedding


class TestEmbeddingSerialization:
    """Embedding の to_bytes / from_bytes のテスト。"""

    def test_roundtrip(self) -> None:
        """シリアライズ→デシリアライズで値が一致する。"""
        original = make_embedding([1.0, 2.0, 3.0])
        restored = Embedding.from_bytes(original.to_bytes())
        np.testing.assert_array_equal(original.values, restored.values)

    def test_dtype_preserved(self) -> None:
        """復元後も dtype が float32 である。"""
        original = make_embedding([1.0, 0.5, -0.5])
        restored = Embedding.from_bytes(original.to_bytes())
        assert restored.values.dtype == np.float32

    def test_empty_bytes_returns_empty_array(self) -> None:
        """空バイトで空配列が返る。"""
        restored = Embedding.from_bytes(b"")
        assert len(restored.values) == 0
        assert restored.values.dtype == np.float32

    def test_restored_is_writable(self) -> None:
        """from_bytes で復元した配列が書き込み可能である。"""
        original = make_embedding([1.0, 2.0, 3.0])
        restored = Embedding.from_bytes(original.to_bytes())
        restored.values[0] = 99.0
        assert restored.values[0] == 99.0


class TestExtractEmbedding:
    """extract_embedding 関数のテスト。"""

    def test_model_path_not_found(self, voiced_audio: AudioData) -> None:
        """不正パスで EmbeddingModelLoadError が発生する。"""
        with pytest.raises(EmbeddingModelLoadError, match="モデルファイルが見つかりません"):
            extract_embedding(voiced_audio, model_path="/nonexistent/path")

    def test_output_dimension(self, voiced_audio_3s: AudioData) -> None:
        """出力ベクトルが192次元である。"""
        result = extract_embedding(voiced_audio_3s)
        assert result.values.shape == (192,)

    def test_output_dtype_is_float32(self, voiced_audio_3s: AudioData) -> None:
        """出力の dtype が float32 である。"""
        result = extract_embedding(voiced_audio_3s)
        assert result.values.dtype == np.float32

    def test_same_audio_produces_same_embedding(self, voiced_audio_3s: AudioData) -> None:
        """同一音声から同一の埋め込みベクトルが得られる。"""
        result1 = extract_embedding(voiced_audio_3s)
        result2 = extract_embedding(voiced_audio_3s)
        np.testing.assert_array_equal(result1.values, result2.values)

    def test_result_type(self, voiced_audio_3s: AudioData) -> None:
        """戻り値が Embedding である。"""
        result = extract_embedding(voiced_audio_3s)
        assert isinstance(result, Embedding)

    def test_empty_audio_raises_error(self) -> None:
        """空音声で EmbeddingExtractionError が発生する。"""
        empty = AudioData(samples=np.array([], dtype=np.int16), sample_rate=16000)
        with pytest.raises(EmbeddingExtractionError, match="空の音声データ"):
            extract_embedding(empty)

    def test_short_audio_raises_error(self) -> None:
        """短すぎる音声で EmbeddingExtractionError が発生する。"""
        # 50ms の極短音声 (100ms 未満)
        short = AudioData(
            samples=np.ones(800, dtype=np.int16),
            sample_rate=16000,
        )
        with pytest.raises(EmbeddingExtractionError, match="短すぎて"):
            extract_embedding(short)
