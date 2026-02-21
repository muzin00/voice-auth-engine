"""汎用数学関数。"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


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
