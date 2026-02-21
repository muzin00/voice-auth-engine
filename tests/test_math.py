"""math モジュールのテスト。"""

import numpy as np
import pytest

from voice_auth_engine.math import cosine_similarity


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
