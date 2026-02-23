"""phoneme_extractor モジュールのテスト。"""

import pytest

from voice_auth_engine.passphrase_validator import EmptyPassphraseError
from voice_auth_engine.phoneme_extractor import (
    Phoneme,
    extract_phonemes,
)


class TestExtractPhonemes:
    """extract_phonemes のテスト。"""

    def test_returns_phoneme(self) -> None:
        """Phoneme を返す。"""
        phoneme = extract_phonemes("こんにちは")
        assert isinstance(phoneme, Phoneme)
        assert len(phoneme.values) > 0
        assert all(isinstance(p, str) for p in phoneme.values)

    def test_unique_count_matches_set(self) -> None:
        """unique_count が unique の要素数と一致する。"""
        phoneme = extract_phonemes("こんにちは世界")
        assert phoneme.unique_count == len(phoneme.unique)

    def test_filters_pau(self) -> None:
        """'pau' が結果に含まれない。"""
        phoneme = extract_phonemes("今日はいい天気ですね")
        assert "pau" not in phoneme.values

    def test_filters_cl(self) -> None:
        """'cl' が結果に含まれない。"""
        phoneme = extract_phonemes("きっと大丈夫")
        assert "cl" not in phoneme.values

    def test_empty_string_raises_error(self) -> None:
        """空文字で EmptyPassphraseError が発生する。"""
        with pytest.raises(EmptyPassphraseError):
            extract_phonemes("")

    def test_whitespace_only_raises_error(self) -> None:
        """空白のみで EmptyPassphraseError が発生する。"""
        with pytest.raises(EmptyPassphraseError):
            extract_phonemes("   ")
