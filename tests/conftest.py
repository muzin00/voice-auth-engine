"""テスト用フィクスチャ。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData

from .audio_factory import (
    generate_audio_file,
    generate_silence_samples,
    generate_voiced_samples,
    make_audio_data,
)


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


@pytest.fixture
def voiced_audio() -> AudioData:
    """1秒の発話風 AudioData。"""
    return make_audio_data(generate_voiced_samples(duration=1.0))


@pytest.fixture
def silence_audio() -> AudioData:
    """1秒の無音 AudioData。"""
    return make_audio_data(generate_silence_samples(duration=1.0))


@pytest.fixture
def empty_audio() -> AudioData:
    """空の AudioData。"""
    return make_audio_data(np.array([], dtype=np.int16))


@pytest.fixture
def voiced_audio_3s() -> AudioData:
    """3秒の発話風 AudioData。"""
    return make_audio_data(generate_voiced_samples(duration=3.0))
