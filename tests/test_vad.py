"""VAD モジュールのテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.vad import (
    SpeechSegment,
    SpeechSegments,
    VadModelLoadError,
    detect_speech,
    extract_speech,
)

from .audio_factory import generate_silence_samples, generate_voiced_samples, make_audio_data
from .conftest import requires_vad_model

SAMPLE_RATE = 16000


@requires_vad_model
class TestDetectSpeech:
    """detect_speech 関数のテスト。"""

    def test_detects_speech_in_voiced_signal(self, voiced_audio: AudioData) -> None:
        """発話風信号で検出される。"""
        result = detect_speech(voiced_audio)
        assert len(result.segments) > 0

    def test_no_speech_in_silence(self, silence_audio: AudioData) -> None:
        """無音で空リスト。"""
        result = detect_speech(silence_audio)
        assert len(result.segments) == 0

    def test_speech_surrounded_by_silence(self) -> None:
        """無音+発話+無音の区間検出。"""
        silence = generate_silence_samples(duration=0.5)
        voiced = generate_voiced_samples(duration=1.0)
        audio = make_audio_data(np.concatenate([silence, voiced, silence]))
        result = detect_speech(audio)
        assert len(result.segments) >= 1
        seg = result.segments[0]
        assert seg.start_sec >= 0.3

    def test_returns_audio_in_result(self, voiced_audio: AudioData) -> None:
        """結果に元音声を含む。"""
        result = detect_speech(voiced_audio)
        assert result.audio is voiced_audio

    def test_audio_shorter_than_window_size(self) -> None:
        """512未満でもクラッシュしない。"""
        audio = make_audio_data(generate_voiced_samples(duration=0.01))  # 160 samples < 512
        result = detect_speech(audio)
        assert isinstance(result, SpeechSegments)

    def test_empty_audio(self, empty_audio: AudioData) -> None:
        """空音声で空リスト。"""
        result = detect_speech(empty_audio)
        assert len(result.segments) == 0

    def test_custom_threshold(self, voiced_audio: AudioData) -> None:
        """threshold パラメータ動作。"""
        result_high = detect_speech(voiced_audio, threshold=0.99)
        result_low = detect_speech(voiced_audio, threshold=0.1)
        assert len(result_high.segments) <= len(result_low.segments)

    def test_segment_timestamps_are_consistent(self, voiced_audio: AudioData) -> None:
        """サンプル↔秒の整合性。"""
        result = detect_speech(voiced_audio)
        for seg in result.segments:
            assert seg.start_sec == pytest.approx(seg.start_sample / SAMPLE_RATE)
            assert seg.end_sec == pytest.approx(seg.end_sample / SAMPLE_RATE)

    def test_segment_end_does_not_exceed_length(self, voiced_audio: AudioData) -> None:
        """範囲外なし。"""
        result = detect_speech(voiced_audio)
        for seg in result.segments:
            assert seg.end_sample <= len(voiced_audio.samples)

    def test_model_path_not_found(self, voiced_audio: AudioData) -> None:
        """不正パスで VadModelLoadError。"""
        with pytest.raises(VadModelLoadError):
            detect_speech(voiced_audio, model_path="/nonexistent/model.onnx")


@requires_vad_model
class TestExtractSpeech:
    """extract_speech 関数のテスト。"""

    def test_extracts_single_segment(self, voiced_audio: AudioData) -> None:
        """単一区間の切り出し。"""
        segments = SpeechSegments(
            segments=[SpeechSegment(1000, 5000, 1000 / SAMPLE_RATE, 5000 / SAMPLE_RATE)],
            audio=voiced_audio,
        )
        result = extract_speech(segments)
        assert len(result.samples) == 4000

    def test_extracts_multiple_segments(self, voiced_audio: AudioData) -> None:
        """複数区間の結合。"""
        segments = SpeechSegments(
            segments=[
                SpeechSegment(1000, 3000, 1000 / SAMPLE_RATE, 3000 / SAMPLE_RATE),
                SpeechSegment(5000, 7000, 5000 / SAMPLE_RATE, 7000 / SAMPLE_RATE),
            ],
            audio=voiced_audio,
        )
        result = extract_speech(segments)
        assert len(result.samples) == 4000

    def test_empty_segments_returns_empty_audio(self, voiced_audio: AudioData) -> None:
        """空セグメント → 空音声。"""
        segments = SpeechSegments(segments=[], audio=voiced_audio)
        result = extract_speech(segments)
        assert len(result.samples) == 0

    def test_output_preserves_sample_rate(self, voiced_audio: AudioData) -> None:
        """サンプルレート維持。"""
        segments = SpeechSegments(
            segments=[SpeechSegment(0, 8000, 0.0, 0.5)],
            audio=voiced_audio,
        )
        result = extract_speech(segments)
        assert result.sample_rate == SAMPLE_RATE

    def test_output_dtype_is_int16(self, voiced_audio: AudioData) -> None:
        """dtype 維持。"""
        segments = SpeechSegments(
            segments=[SpeechSegment(0, 8000, 0.0, 0.5)],
            audio=voiced_audio,
        )
        result = extract_speech(segments)
        assert result.samples.dtype == np.int16

    def test_roundtrip_detect_then_extract(self, voiced_audio: AudioData) -> None:
        """detect → extract の統合テスト。"""
        segments = detect_speech(voiced_audio)
        result = extract_speech(segments)
        assert len(result.samples) > 0
        assert result.sample_rate == SAMPLE_RATE
        assert result.samples.dtype == np.int16
