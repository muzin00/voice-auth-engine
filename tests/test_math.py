"""math モジュールのテスト。"""

import numpy as np
import pytest

from voice_auth_engine.math import cosine_similarity, normalized_edit_distance


class TestCosineSimilarity:
    """cosine_similarity のテスト。"""

    def test_identical_vectors(self) -> None:
        """同一ベクトル → 1.0。"""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """直交ベクトル → 0.0。"""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """逆向きベクトル → -1.0。"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        """ゼロベクトル → 0.0。"""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        z = np.zeros(3, dtype=np.float32)
        assert cosine_similarity(a, z) == pytest.approx(0.0)
        assert cosine_similarity(z, a) == pytest.approx(0.0)


class TestNormalizedEditDistance:
    """normalized_edit_distance のテスト。"""

    def test_identical_sequences(self) -> None:
        """同一系列 → 0.0。"""
        seq = ["a", "b", "c"]
        assert normalized_edit_distance(seq, seq) == pytest.approx(0.0)

    def test_both_empty(self) -> None:
        """両方空 → 0.0。"""
        assert normalized_edit_distance([], []) == pytest.approx(0.0)

    def test_one_empty(self) -> None:
        """片方空 → 1.0。"""
        assert normalized_edit_distance(["a", "b"], []) == pytest.approx(1.0)
        assert normalized_edit_distance([], ["a", "b"]) == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        """部分一致 → 1/3。"""
        a = ["a", "b", "c"]
        b = ["a", "b", "d"]
        assert normalized_edit_distance(a, b) == pytest.approx(1 / 3)

    def test_completely_different(self) -> None:
        """完全不一致 → 1.0。"""
        a = ["a", "b", "c"]
        b = ["x", "y", "z"]
        assert normalized_edit_distance(a, b) == pytest.approx(1.0)
