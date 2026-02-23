"""日本語パスフレーズの音素多様性バリデーション。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from voice_auth_engine.math import pairwise_distances

if TYPE_CHECKING:
    from voice_auth_engine.phoneme_extractor import Phoneme


class PassphraseValidationError(Exception):
    """パスフレーズバリデーションの基底例外。"""


class EmptyPassphraseError(PassphraseValidationError):
    """パスフレーズが空。"""


class InsufficientPhonemeError(PassphraseValidationError):
    """ユニーク音素数が不足。"""

    def __init__(self, phoneme: Phoneme, min_required: int) -> None:
        self.phoneme = phoneme
        self.min_required = min_required
        super().__init__(f"ユニーク音素数が不足しています: {phoneme.unique_count} < {min_required}")


class PhonemeConsistencyError(PassphraseValidationError):
    """音素列の整合性チェック失敗。"""


def validate_passphrase(phoneme: Phoneme, *, min_unique_phonemes: int) -> None:
    """音素解析結果のユニーク音素数を検証する。

    Args:
        phoneme: 検証対象の音素解析結果。
        min_unique_phonemes: 必要な最小ユニーク音素数。

    Raises:
        InsufficientPhonemeError: ユニーク音素数が不足の場合。
    """
    if phoneme.unique_count < min_unique_phonemes:
        raise InsufficientPhonemeError(phoneme, min_unique_phonemes)


def validate_phoneme_consistency(samples: list[Phoneme], *, threshold: float) -> None:
    """音素サンプル間の整合性を検証する。

    全ペア間の正規化編集距離が閾値以下であることを確認する。

    Args:
        samples: 音素サンプルのリスト。
        threshold: 許容する正規化編集距離の閾値。

    Raises:
        PhonemeConsistencyError: いずれかのペアの距離が閾値を超えた場合。
    """
    distances = pairwise_distances([s.values for s in samples])
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            if distances[i][j] > threshold:
                raise PhonemeConsistencyError(
                    f"音素列の不整合: サンプル {i} と {j} の距離 "
                    f"{distances[i][j]:.3f} が閾値 {threshold} を超えています"
                )
