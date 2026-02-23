"""汎用数学関数。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T")


def cosine_similarity(
    a: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
) -> float:
    """2つのベクトルのコサイン類似度を計算する。

    Args:
        a: ベクトル a。
        b: ベクトル b。a と同じ次元数。

    Returns:
        コサイン類似度 [-1.0, 1.0]。
        いずれかがゼロベクトルの場合は 0.0。
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalized_edit_distance(a: Sequence[T], b: Sequence[T]) -> float:
    """2つの系列の正規化編集距離を計算する。

    レーベンシュタイン距離を max(len(a), len(b)) で正規化する。
    Wagner-Fischer DP アルゴリズム（省メモリ 1行DP）で実装。

    Args:
        a: 系列 a。
        b: 系列 b。

    Returns:
        正規化編集距離 [0.0, 1.0]。0.0 = 完全一致。両方空なら 0.0。
    """
    len_a, len_b = len(a), len(b)
    if len_a == 0 and len_b == 0:
        return 0.0

    # 短い方を row にして省メモリ化
    if len_a > len_b:
        a, b = b, a
        len_a, len_b = len_b, len_a

    prev = list(range(len_a + 1))
    for j in range(1, len_b + 1):
        curr = [j] + [0] * len_a
        for i in range(1, len_a + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[i] = min(
                curr[i - 1] + 1,  # 挿入
                prev[i] + 1,  # 削除
                prev[i - 1] + cost,  # 置換
            )
        prev = curr

    return prev[len_a] / max(len_a, len_b)


def pairwise_distances(
    sequences: Sequence[Sequence[T]],
    distance_fn: Callable[[Sequence[T], Sequence[T]], float] = normalized_edit_distance,
) -> list[list[float]]:
    """全ペア間の距離行列を計算する。

    Returns:
        n×n の対称行列。distances[i][j] は sequences[i] と sequences[j] の距離。
    """
    n = len(sequences)
    distances = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_fn(sequences[i], sequences[j])
            distances[i][j] = d
            distances[j][i] = d
    return distances


def select_medoid(distances: Sequence[Sequence[float]]) -> int:
    """距離行列からメドイド（他の全要素との距離合計が最小の要素）のインデックスを返す。

    同スコアの場合は最小インデックスを採用。
    """
    min_index = 0
    min_sum = float("inf")
    for i, row in enumerate(distances):
        total = sum(row)
        if total < min_sum:
            min_sum = total
            min_index = i
    return min_index
