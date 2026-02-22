"""audio_preprocessor モジュールのテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import (
    AudioData,
    AudioDecodeError,
    UnsupportedFormatError,
    load_audio,
    load_audio_bytes,
)

from .audio_factory import generate_audio_file

SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".webm", ".aac", ".flac", ".m4a"}


class TestLoadAudio:
    """load_audio 関数のテスト。"""

    def test_loads_wav_file(self, wav_16k_mono: Path) -> None:
        """WAV ファイルを読み込める。"""
        result = load_audio(wav_16k_mono)
        assert len(result.samples) > 0

    def test_loads_flac_file(self, flac_file: Path) -> None:
        """FLAC ファイルを読み込める。"""
        result = load_audio(flac_file)
        assert len(result.samples) > 0

    def test_loads_webm_file(self, webm_file: Path) -> None:
        """WebM (Opus) ファイルを読み込める。"""
        result = load_audio(webm_file)
        assert len(result.samples) > 0

    def test_resamples_to_16000(self, wav_44k_mono: Path) -> None:
        """44.1kHz のファイルを 16kHz にリサンプリングする。"""
        result = load_audio(wav_44k_mono)
        assert result.sample_rate == 16000
        # 0.5 秒 → 約 8000 サンプル (16kHz)
        expected_samples = int(16000 * 0.5)
        assert abs(len(result.samples) - expected_samples) < 100

    def test_converts_stereo_to_mono(self, wav_44k_stereo: Path) -> None:
        """ステレオをモノラルに変換する。"""
        result = load_audio(wav_44k_stereo)
        assert result.samples.ndim == 1

    def test_output_dtype_is_int16(self, wav_16k_mono: Path) -> None:
        """出力の dtype が int16 である。"""
        result = load_audio(wav_16k_mono)
        assert result.samples.dtype == np.int16

    def test_output_sample_rate_is_16000(self, wav_16k_mono: Path) -> None:
        """出力のサンプルレートが 16000 である。"""
        result = load_audio(wav_16k_mono)
        assert result.sample_rate == 16000

    def test_passthrough_16k_mono(self, wav_16k_mono: Path) -> None:
        """16kHz モノラルはそのまま通る。"""
        result = load_audio(wav_16k_mono)
        assert result.sample_rate == 16000
        expected_samples = int(16000 * 0.5)
        assert abs(len(result.samples) - expected_samples) < 100

    def test_raises_file_not_found(self, tmp_audio_dir: Path) -> None:
        """存在しないファイルで FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_audio_dir / "nonexistent.wav")

    def test_raises_unsupported_format(self, tmp_audio_dir: Path) -> None:
        """非対応拡張子で UnsupportedFormatError。"""
        txt_file = tmp_audio_dir / "test.txt"
        txt_file.write_text("not audio")
        with pytest.raises(UnsupportedFormatError):
            load_audio(txt_file)

    def test_raises_audio_decode_error_for_corrupt(self, tmp_audio_dir: Path) -> None:
        """破損ファイルで AudioDecodeError。"""
        corrupt = tmp_audio_dir / "corrupt.wav"
        corrupt.write_bytes(b"not a real audio file")
        with pytest.raises(AudioDecodeError):
            load_audio(corrupt)

    def test_accepts_string_path(self, wav_16k_mono: Path) -> None:
        """str パスを受け付ける。"""
        result = load_audio(str(wav_16k_mono))
        assert len(result.samples) > 0

    def test_case_insensitive_extension(self, tmp_audio_dir: Path) -> None:
        """大文字拡張子を受け付ける。"""
        src = generate_audio_file(tmp_audio_dir / "test_temp.wav")
        upper = tmp_audio_dir / "test.WAV"
        upper.write_bytes(src.read_bytes())
        result = load_audio(upper)
        assert len(result.samples) > 0


class TestLoadAudioBytes:
    """load_audio_bytes 関数のテスト。"""

    def test_decodes_wav_bytes(self, wav_16k_mono: Path) -> None:
        """WAV bytes をデコードできる。"""
        data = wav_16k_mono.read_bytes()
        result = load_audio_bytes(data)
        assert result.sample_rate == 16000
        assert result.samples.dtype == np.int16
        assert len(result.samples) > 0

    def test_decodes_with_explicit_format(self, wav_16k_mono: Path) -> None:
        """format 明示指定でデコードできる。"""
        data = wav_16k_mono.read_bytes()
        result = load_audio_bytes(data, format="wav")
        assert result.sample_rate == 16000
        assert len(result.samples) > 0

    def test_raises_on_empty_bytes(self) -> None:
        """空 bytes で AudioDecodeError。"""
        with pytest.raises(AudioDecodeError):
            load_audio_bytes(b"")

    def test_raises_on_invalid_bytes(self) -> None:
        """不正 bytes で AudioDecodeError。"""
        with pytest.raises(AudioDecodeError):
            load_audio_bytes(b"not a real audio file")


class TestAudioData:
    """AudioData の samples_f32 プロパティのテスト。"""

    def test_samples_f32_dtype(self) -> None:
        """float32 の配列を返す。"""
        audio = AudioData(samples=np.array([0, 16384, -16384], dtype=np.int16), sample_rate=16000)
        assert audio.samples_f32.dtype == np.float32
