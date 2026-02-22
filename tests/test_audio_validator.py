"""audio_validator モジュールのテスト。"""

from pathlib import Path

import pytest

from voice_auth_engine.audio_validator import (
    EmptyAudioError,
    InsufficientDurationError,
    UnsupportedExtensionError,
    validate_audio,
    validate_extension,
)


class TestValidateAudio:
    """validate_audio のテスト。"""

    def test_passes_with_sufficient_duration(self, voiced_audio) -> None:
        """十分な発話時間で正常に通過する。"""
        validate_audio(voiced_audio, min_seconds=0.5)

    def test_raises_with_insufficient_duration(self, voiced_audio) -> None:
        """発話時間不足で InsufficientDurationError が発生する。"""
        with pytest.raises(InsufficientDurationError):
            validate_audio(voiced_audio, min_seconds=10.0)

    def test_empty_audio_raises_error(self, empty_audio) -> None:
        """空音声で EmptyAudioError が発生する。"""
        with pytest.raises(EmptyAudioError):
            validate_audio(empty_audio)

    def test_error_contains_attributes(self, voiced_audio) -> None:
        """例外に duration_seconds と min_seconds が含まれる。"""
        with pytest.raises(InsufficientDurationError) as exc_info:
            validate_audio(voiced_audio, min_seconds=10.0)
        err = exc_info.value
        assert err.duration_seconds > 0
        assert err.min_seconds == 10.0

    def test_boundary_exact_minimum(self, voiced_audio_3s) -> None:
        """境界値（ちょうど閾値）で通過する。"""
        duration = len(voiced_audio_3s.samples) / voiced_audio_3s.sample_rate
        validate_audio(voiced_audio_3s, min_seconds=duration)

    def test_boundary_just_below_raises(self, voiced_audio) -> None:
        """境界値（閾値をわずかに超える min_seconds）でエラーになる。"""
        duration = len(voiced_audio.samples) / voiced_audio.sample_rate
        with pytest.raises(InsufficientDurationError):
            validate_audio(voiced_audio, min_seconds=duration + 0.001)


class TestValidateExtension:
    """validate_extension のテスト。"""

    @pytest.mark.parametrize(
        "ext",
        [".wav", ".mp3", ".ogg", ".webm", ".aac", ".flac", ".m4a"],
    )
    def test_supported_extensions(self, ext: str) -> None:
        """サポート対象の拡張子で正規化された拡張子が返る。"""
        result = validate_extension(f"audio{ext}")
        assert result == ext

    def test_unsupported_extension_raises_error(self) -> None:
        """非対応拡張子で UnsupportedExtensionError が発生する。"""
        with pytest.raises(UnsupportedExtensionError):
            validate_extension("audio.txt")

    def test_error_contains_extension(self) -> None:
        """例外に extension が含まれる。"""
        with pytest.raises(UnsupportedExtensionError) as exc_info:
            validate_extension("audio.txt")
        assert exc_info.value.extension == ".txt"

    def test_uppercase_extension(self) -> None:
        """大文字拡張子が正規化される。"""
        result = validate_extension("audio.WAV")
        assert result == ".wav"

    def test_mixed_case_extension(self) -> None:
        """大文字小文字混在の拡張子が正規化される。"""
        result = validate_extension("audio.Mp3")
        assert result == ".mp3"

    def test_path_object(self) -> None:
        """Path オブジェクトを受け付ける。"""
        result = validate_extension(Path("dir/audio.flac"))
        assert result == ".flac"
