"""日本語パスフレーズの音素多様性バリデーション。"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
