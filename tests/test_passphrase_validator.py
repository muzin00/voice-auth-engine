"""passphrase_validator モジュールのテスト。"""

import pytest

from voice_auth_engine.passphrase_validator import (
    EmptyPassphraseError,
    InsufficientPhonemeError,
    PassphraseInfo,
    analyze_passphrase,
    validate_passphrase,
)


class TestAnalyzePassphrase:
    """analyze_passphrase のテスト。"""

    def test_extracts_phonemes(self) -> None:
        """日本語テキストから音素リストを取得できる。"""
        info = analyze_passphrase("こんにちは")
        assert len(info.phonemes) > 0
        assert all(isinstance(p, str) for p in info.phonemes)

    def test_filters_pau(self) -> None:
        """'pau' が結果に含まれない。"""
        info = analyze_passphrase("今日はいい天気ですね")
        assert "pau" not in info.phonemes

    def test_filters_cl(self) -> None:
        """'cl' が結果に含まれない。"""
        info = analyze_passphrase("きっと大丈夫")
        assert "cl" not in info.phonemes

    def test_unique_count_matches_set(self) -> None:
        """unique_count が unique_phonemes の要素数と一致する。"""
        info = analyze_passphrase("こんにちは世界")
        assert info.unique_count == len(info.unique_phonemes)

    def test_preserves_original_text(self) -> None:
        """info.text が入力テキストと一致する。"""
        text = "おはようございます"
        info = analyze_passphrase(text)
        assert info.text == text

    def test_empty_string_raises_error(self) -> None:
        """空文字で EmptyPassphraseError が発生する。"""
        with pytest.raises(EmptyPassphraseError):
            analyze_passphrase("")

    def test_whitespace_only_raises_error(self) -> None:
        """空白のみで EmptyPassphraseError が発生する。"""
        with pytest.raises(EmptyPassphraseError):
            analyze_passphrase("   ")


class TestValidatePassphrase:
    """validate_passphrase のテスト。"""

    def test_passes_with_sufficient_phonemes(self) -> None:
        """十分な音素数で正常に返却される。"""
        info = validate_passphrase("今日はいい天気ですね")
        assert isinstance(info, PassphraseInfo)
        assert info.unique_count >= 5

    def test_raises_with_insufficient_phonemes(self) -> None:
        """音素不足で InsufficientPhonemeError が発生する。"""
        with pytest.raises(InsufficientPhonemeError):
            validate_passphrase("あ", min_unique_phonemes=10)

    def test_custom_min_unique_phonemes(self) -> None:
        """カスタム閾値での動作を確認する。"""
        info = validate_passphrase("こんにちは", min_unique_phonemes=1)
        assert info.unique_count >= 1

    def test_error_contains_info(self) -> None:
        """例外に info と min_required が含まれる。"""
        with pytest.raises(InsufficientPhonemeError) as exc_info:
            validate_passphrase("あ", min_unique_phonemes=100)
        err = exc_info.value
        assert isinstance(err.info, PassphraseInfo)
        assert err.min_required == 100

    def test_boundary_exact_minimum(self) -> None:
        """境界値（ちょうど閾値）で通過する。"""
        info = analyze_passphrase("こんにちは")
        exact = info.unique_count
        result = validate_passphrase("こんにちは", min_unique_phonemes=exact)
        assert result.unique_count == exact
