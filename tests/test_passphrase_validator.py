"""passphrase_validator モジュールのテスト。"""

import pytest

from voice_auth_engine.passphrase_validator import (
    InsufficientPhonemeError,
    validate_passphrase,
)
from voice_auth_engine.phoneme_extractor import Phoneme


class TestValidatePassphrase:
    """validate_passphrase のテスト。"""

    def test_passes_with_sufficient_phonemes(self) -> None:
        """十分なユニーク音素数で例外が発生しない。"""
        phoneme = Phoneme(values=["a", "i", "u", "e", "o"])
        validate_passphrase(phoneme, min_unique_phonemes=5)

    def test_raises_with_insufficient_phonemes(self) -> None:
        """ユニーク音素数不足で InsufficientPhonemeError が発生する。"""
        phoneme = Phoneme(values=["a", "i"])
        with pytest.raises(InsufficientPhonemeError):
            validate_passphrase(phoneme, min_unique_phonemes=5)

    def test_error_contains_phoneme(self) -> None:
        """例外に phoneme と min_required が含まれる。"""
        phoneme = Phoneme(values=["a"])
        with pytest.raises(InsufficientPhonemeError) as exc_info:
            validate_passphrase(phoneme, min_unique_phonemes=100)
        err = exc_info.value
        assert err.phoneme is phoneme
        assert err.min_required == 100

    def test_boundary_exact_minimum(self) -> None:
        """境界値（ちょうど閾値）で通過する。"""
        phoneme = Phoneme(values=["a", "i", "u", "e", "o"])
        validate_passphrase(phoneme, min_unique_phonemes=5)

    def test_boundary_one_below_minimum(self) -> None:
        """境界値（閾値-1）で例外が発生する。"""
        phoneme = Phoneme(values=["a", "i", "u", "e"])
        with pytest.raises(InsufficientPhonemeError):
            validate_passphrase(phoneme, min_unique_phonemes=5)
