"""math モジュールのテスト。"""

import numpy as np
import pytest

from voice_auth_engine.math import (
    cosine_similarity,
    normalized_edit_distance,
    pairwise_distances,
    select_medoid,
)


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


class TestPairwiseDistances:
    """pairwise_distances のテスト。"""

    def test_normal(self) -> None:
        """正常系: 距離行列が対称であること。"""
        seqs = [["a", "b", "c"], ["a", "b", "d"], ["x", "y", "z"]]
        result = pairwise_distances(seqs)

        assert len(result) == 3
        # 対角は 0
        for i in range(3):
            assert result[i][i] == pytest.approx(0.0)
        # 対称
        for i in range(3):
            for j in range(3):
                assert result[i][j] == pytest.approx(result[j][i])
        # 具体値
        assert result[0][1] == pytest.approx(1 / 3)
        assert result[0][2] == pytest.approx(1.0)

    def test_single_element(self) -> None:
        """単一要素 → 1×1 のゼロ行列。"""
        result = pairwise_distances([["a", "b"]])
        assert result == [[0.0]]

    def test_empty_list(self) -> None:
        """空リスト → 空行列。"""
        result = pairwise_distances([])
        assert result == []


class TestSelectMedoid:
    """select_medoid のテスト。"""

    def test_normal(self) -> None:
        """正常系: 距離合計が最小の要素を返す。"""
        # idx0: 0+1+3=4, idx1: 1+0+1=2, idx2: 3+1+0=4
        distances = [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 1.0],
            [3.0, 1.0, 0.0],
        ]
        assert select_medoid(distances) == 1

    def test_tie_returns_first(self) -> None:
        """同スコア時は最小インデックスを返す。"""
        # idx0: 0+1=1, idx1: 1+0=1
        distances = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]
        assert select_medoid(distances) == 0

    def test_single_element(self) -> None:
        """単一要素 → 0。"""
        assert select_medoid([[0.0]]) == 0
