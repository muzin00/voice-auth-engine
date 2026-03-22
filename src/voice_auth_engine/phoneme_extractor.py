"""pyopenjtalk を使用した日本語テキストの音素抽出。"""

from __future__ import annotations

import pyopenjtalk

from voice_auth_engine.phoneme_validator import EmptyPhonemeError

# フィルタ対象の音素記号
_FILTERED_PHONEMES: frozenset[str] = frozenset({"pau", "cl"})


class Phoneme:
    """音素解析結果。"""

    def __init__(self, values: list[str]) -> None:
        self.values = values

    @property
    def unique(self) -> set[str]:
        """ユニーク音素の集合。"""
        return set(self.values)

    @property
    def unique_count(self) -> int:
        """ユニーク音素数。"""
        return len(self.unique)


def extract_phonemes(text: str) -> Phoneme:
    """テキストから音素を抽出する。

    Args:
        text: 入力テキスト。

    Returns:
        Phoneme: 音素解析結果。

    Raises:
        EmptyPhonemeError: テキストが空または空白のみの場合。
    """
    if not text.strip():
        raise EmptyPhonemeError("テキストが空です")

    raw_phonemes: list[str] = pyopenjtalk.g2p(text, join=False)
    phonemes = [p for p in raw_phonemes if p not in _FILTERED_PHONEMES]

    return Phoneme(values=phonemes)
