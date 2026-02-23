"""speech_detector モジュールの統合テスト（実音声）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData, load_audio
from voice_auth_engine.speech_detector import detect_speech, extract_speech

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RATE = 16000


@pytest.fixture
def digits_clear() -> AudioData:
    """数字を連続で読み上げた音声。"""
    return load_audio(FIXTURES_DIR / "digits_clear.mp3")


@pytest.fixture
def digits_with_pauses() -> AudioData:
    """数字間にポーズを挟んだ音声。"""
    return load_audio(FIXTURES_DIR / "digits_with_pauses.mp3")


@pytest.fixture
def digit_single_short() -> AudioData:
    """1桁の短い音声。"""
    return load_audio(FIXTURES_DIR / "digit_single_short.mp3")


class TestDetectSpeechWithRealAudio:
    """実音声を使った detect_speech のテスト。"""

    def test_detects_speech_in_clear_digits(self, digits_clear: AudioData) -> None:
        """明瞭な数字読み上げで発話が検出される。"""
        result = detect_speech(digits_clear)
        assert len(result.segments) >= 1

    def test_segments_within_audio_bounds(self, digits_clear: AudioData) -> None:
        """検出区間が音声長を超えない。"""
        result = detect_speech(digits_clear)
        for seg in result.segments:
            assert 0 <= seg.start_sample < seg.end_sample <= len(digits_clear.samples)
            assert seg.start_sec < seg.end_sec

    def test_timestamps_consistent_with_samples(self, digits_clear: AudioData) -> None:
        """サンプル位置と秒数が一致する。"""
        result = detect_speech(digits_clear)
        for seg in result.segments:
            assert seg.start_sec == pytest.approx(seg.start_sample / SAMPLE_RATE)
            assert seg.end_sec == pytest.approx(seg.end_sample / SAMPLE_RATE)

    def test_detects_speech_in_paused_audio(self, digits_with_pauses: AudioData) -> None:
        """ポーズ入り音声で発話が検出される。"""
        result = detect_speech(digits_with_pauses)
        assert len(result.segments) >= 1

    def test_segments_are_chronologically_ordered(self, digits_with_pauses: AudioData) -> None:
        """セグメントが時系列順。"""
        result = detect_speech(digits_with_pauses)
        for i in range(len(result.segments) - 1):
            assert result.segments[i].end_sample <= result.segments[i + 1].start_sample

    def test_short_digit_detected_with_lower_min_duration(
        self, digit_single_short: AudioData
    ) -> None:
        """短い1桁の発話は min_speech_duration を下げると検出される。"""
        result = detect_speech(digit_single_short, min_speech_duration=0.1)
        assert len(result.segments) >= 1

    def test_extracted_speech_has_nonzero_amplitude(self, digits_clear: AudioData) -> None:
        """抽出した発話音声が実際の音声データを含む（ゼロでない）。"""
        segments = detect_speech(digits_clear)
        speech = extract_speech(segments)
        rms = np.sqrt(np.mean(speech.samples.astype(np.float32) ** 2))
        assert rms > 0

    def test_detected_speech_duration_is_reasonable(self, digits_clear: AudioData) -> None:
        """検出された発話時間が音声の実際の内容に見合う長さである。"""
        segments = detect_speech(digits_clear)
        total_speech_sec = sum(s.end_sec - s.start_sec for s in segments.segments)
        assert total_speech_sec >= 1.0

    def test_quiet_speech_detected(self, digits_clear: AudioData) -> None:
        """音量を下げた音声でも検出される。"""
        quiet_samples = (digits_clear.samples.astype(np.float32) * 0.15).astype(np.int16)
        quiet_audio = AudioData(samples=quiet_samples, sample_rate=SAMPLE_RATE)
        result = detect_speech(quiet_audio)
        assert len(result.segments) >= 1

    def test_with_leading_trailing_silence(self, digits_clear: AudioData) -> None:
        """前後に無音を付加しても発話区間が検出される。"""
        silence = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)
        padded_samples = np.concatenate([silence, digits_clear.samples, silence])
        padded_audio = AudioData(samples=padded_samples, sample_rate=SAMPLE_RATE)
        result = detect_speech(padded_audio)
        assert len(result.segments) >= 1
        assert result.segments[0].start_sec >= 1.5


class TestExtractSpeechWithRealAudio:
    """実音声を使った extract_speech のテスト。"""

    def test_extract_preserves_format(self, digits_clear: AudioData) -> None:
        """抽出結果のフォーマットが正しい。"""
        segments = detect_speech(digits_clear)
        result = extract_speech(segments)
        assert result.sample_rate == SAMPLE_RATE
        assert result.samples.dtype == np.int16
        assert len(result.samples) > 0

    def test_extracted_length_less_than_original(self, digits_with_pauses: AudioData) -> None:
        """ポーズ入り音声で抽出結果が元音声より短い。"""
        segments = detect_speech(digits_with_pauses)
        result = extract_speech(segments)
        assert len(result.samples) < len(digits_with_pauses.samples)

    def test_roundtrip_with_real_audio(self) -> None:
        """load_audio → detect_speech → extract_speech の全パイプライン。"""
        audio = load_audio(FIXTURES_DIR / "digits_clear.mp3")
        segments = detect_speech(audio)
        result = extract_speech(segments)
        assert len(result.samples) > 0
        assert result.sample_rate == SAMPLE_RATE
        assert result.samples.dtype == np.int16
