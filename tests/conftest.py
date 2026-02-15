"""テスト用フィクスチャ。"""

from __future__ import annotations

from pathlib import Path

import pytest

from .audio_factory import generate_audio_file


@pytest.fixture
def tmp_audio_dir(tmp_path: Path) -> Path:
    """一時音声ファイル用ディレクトリ。"""
    return tmp_path


@pytest.fixture
def wav_16k_mono(tmp_audio_dir: Path) -> Path:
    """16kHz モノラル WAV ファイル。"""
    return generate_audio_file(tmp_audio_dir / "test_16k_mono.wav")


@pytest.fixture
def wav_44k_mono(tmp_audio_dir: Path) -> Path:
    """44.1kHz モノラル WAV ファイル。"""
    return generate_audio_file(
        tmp_audio_dir / "test_44k_mono.wav",
        sample_rate=44100,
    )


@pytest.fixture
def wav_44k_stereo(tmp_audio_dir: Path) -> Path:
    """44.1kHz ステレオ WAV ファイル。"""
    return generate_audio_file(
        tmp_audio_dir / "test_44k_stereo.wav",
        sample_rate=44100,
        channels=2,
    )


@pytest.fixture
def flac_file(tmp_audio_dir: Path) -> Path:
    """FLAC ファイル。"""
    return generate_audio_file(
        tmp_audio_dir / "test.flac",
        codec="flac",
        format_name="flac",
    )


@pytest.fixture
def webm_file(tmp_audio_dir: Path) -> Path:
    """WebM (Opus) ファイル。"""
    return generate_audio_file(
        tmp_audio_dir / "test.webm",
        sample_rate=48000,
        codec="libopus",
        format_name="webm",
    )
