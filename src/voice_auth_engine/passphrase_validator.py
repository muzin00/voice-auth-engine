"""pyopenjtalk を使用した日本語パスフレーズの音素多様性バリデーション。"""

from __future__ import annotations

from typing import NamedTuple

import pyopenjtalk

# フィルタ対象の音素記号
_FILTERED_PHONEMES: frozenset[str] = frozenset({"pau", "cl"})


class PassphraseValidationError(Exception):
    """パスフレーズバリデーションの基底例外。"""


class EmptyPassphraseError(PassphraseValidationError):
    """パスフレーズが空。"""


class InsufficientPhonemeError(PassphraseValidationError):
    """ユニーク音素数が不足。"""

    def __init__(self, info: PassphraseInfo, min_required: int) -> None:
        self.info = info
        self.min_required = min_required
        super().__init__(f"ユニーク音素数が不足しています: {info.unique_count} < {min_required}")


class PassphraseInfo(NamedTuple):
    """パスフレーズの音素解析結果。"""

    text: str
    phonemes: list[str]
    unique_phonemes: set[str]
    unique_count: int


def analyze_passphrase(text: str) -> PassphraseInfo:
    """パスフレーズの音素を解析する（バリデーションなし）。

    Args:
        text: パスフレーズのテキスト。

    Returns:
        PassphraseInfo: 音素解析結果。

    Raises:
        EmptyPassphraseError: テキストが空または空白のみの場合。
    """
    if not text.strip():
        raise EmptyPassphraseError("パスフレーズが空です")

    raw_phonemes: list[str] = pyopenjtalk.g2p(text, join=False)
    phonemes = [p for p in raw_phonemes if p not in _FILTERED_PHONEMES]
    unique_phonemes = set(phonemes)

    return PassphraseInfo(
        text=text,
        phonemes=phonemes,
        unique_phonemes=unique_phonemes,
        unique_count=len(unique_phonemes),
    )


def validate_passphrase(
    text: str,
    *,
    min_unique_phonemes: int = 5,
) -> PassphraseInfo:
    """パスフレーズの音素多様性を検証する。

    analyze_passphrase で解析した後、ユニーク音素数が
    min_unique_phonemes 以上であることを確認する。

    Args:
        text: パスフレーズのテキスト。
        min_unique_phonemes: 必要な最小ユニーク音素数。

    Returns:
        PassphraseInfo: バリデーション通過時の音素解析結果。

    Raises:
        EmptyPassphraseError: テキストが空または空白のみの場合。
        InsufficientPhonemeError: ユニーク音素数が不足の場合。
    """
    info = analyze_passphrase(text)

    if info.unique_count < min_unique_phonemes:
        raise InsufficientPhonemeError(info, min_unique_phonemes)

    return info
