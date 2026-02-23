"""speech_recognizer モジュールのテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.speech_recognizer import (
    RecognitionError,
    RecognizerModelLoadError,
    TranscriptionResult,
    transcribe,
)


class TestTranscribe:
    """transcribe 関数のテスト。"""

    def test_model_dir_not_found(self, voiced_audio: AudioData) -> None:
        """不正パスで RecognizerModelLoadError が発生する。"""
        with pytest.raises(RecognizerModelLoadError, match="ディレクトリが見つかりません"):
            transcribe(voiced_audio, model_dir="/nonexistent/path")

    def test_model_file_missing(
        self, voiced_audio: AudioData, tmp_path: pytest.TempPathFactory
    ) -> None:
        """ディレクトリはあるが onnx ファイルがない場合。"""
        with pytest.raises(RecognizerModelLoadError, match="モデルファイルが見つかりません"):
            transcribe(voiced_audio, model_dir=str(tmp_path))

    def test_transcribes_voiced_audio(self, voiced_audio: AudioData) -> None:
        """発話風音声で空でないテキストを返す。"""
        result = transcribe(voiced_audio)
        assert isinstance(result.text, str)
        assert len(result.text) > 0

    def test_result_type(self, voiced_audio: AudioData) -> None:
        """戻り値が TranscriptionResult である。"""
        result = transcribe(voiced_audio)
        assert isinstance(result, TranscriptionResult)

    def test_empty_audio_raises_error(self) -> None:
        """空音声で RecognitionError が発生する。"""
        empty = AudioData(samples=np.array([], dtype=np.int16), sample_rate=16000)
        with pytest.raises(RecognitionError, match="空の音声データ"):
            transcribe(empty)

    def test_custom_language(self, voiced_audio: AudioData) -> None:
        """language パラメータが受け付けられる。"""
        result = transcribe(voiced_audio, language="en")
        assert isinstance(result, TranscriptionResult)
