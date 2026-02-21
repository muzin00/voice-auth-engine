"""speaker_verifier モジュールのテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_auth_engine.speaker_verifier import (
    InsufficientSpeechError,
    PassphraseEnroller,
    PassphraseVerifier,
    check_speech_duration,
)

from .audio_factory import make_audio, make_embedding


class TestCheckSpeechDuration:
    """check_speech_duration のテスト。"""

    def test_sufficient_duration(self) -> None:
        """十分な長さで例外なし。"""
        audio = make_audio(5.0)
        check_speech_duration(audio, min_seconds=3.0)

    def test_insufficient_duration(self) -> None:
        """不足で InsufficientSpeechError。"""
        audio = make_audio(1.0)
        with pytest.raises(InsufficientSpeechError):
            check_speech_duration(audio, min_seconds=3.0)

    def test_exact_boundary(self) -> None:
        """境界値（ちょうど min_seconds）で通過。"""
        audio = make_audio(3.0)
        check_speech_duration(audio, min_seconds=3.0)

    def test_custom_min_seconds(self) -> None:
        """カスタム閾値での動作。"""
        audio = make_audio(2.0)
        check_speech_duration(audio, min_seconds=1.0)
        with pytest.raises(InsufficientSpeechError):
            check_speech_duration(audio, min_seconds=5.0)


class TestPassphraseEnroller:
    """PassphraseEnroller のテスト。"""

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    def test_add_sample_increments_count(self, mock_extract: MagicMock) -> None:
        """add_sample 後に sample_count が増加する。"""
        mock_extract.return_value = make_embedding([1.0, 0.0, 0.0])
        enroller = PassphraseEnroller(min_speech_seconds=0.1, min_unique_phonemes=None)
        audio = make_audio(1.0)
        enroller.add_sample(audio)
        assert enroller.sample_count == 1
        enroller.add_sample(audio)
        assert enroller.sample_count == 2

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    def test_enroll_returns_verifier(self, mock_extract: MagicMock) -> None:
        """enroll() が PassphraseVerifier を返す。"""
        mock_extract.return_value = make_embedding([1.0, 0.0, 0.0])
        enroller = PassphraseEnroller(min_speech_seconds=0.1, min_unique_phonemes=None)
        enroller.add_sample(make_audio(1.0))
        verifier = enroller.enroll()
        assert isinstance(verifier, PassphraseVerifier)

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    def test_enroll_averages_embeddings(self, mock_extract: MagicMock) -> None:
        """複数サンプルの平均ベクトルで Verifier が生成される。"""
        mock_extract.side_effect = [
            make_embedding([1.0, 0.0, 0.0]),
            make_embedding([0.0, 1.0, 0.0]),
        ]
        enroller = PassphraseEnroller(min_speech_seconds=0.1, min_unique_phonemes=None)
        enroller.add_sample(make_audio(1.0))
        enroller.add_sample(make_audio(1.0))
        verifier = enroller.enroll()
        expected = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(verifier._embedding.values, expected)

    def test_enroll_no_samples_raises(self) -> None:
        """サンプル未蓄積で ValueError。"""
        enroller = PassphraseEnroller(min_unique_phonemes=None)
        with pytest.raises(ValueError):
            enroller.enroll()

    def test_sample_count_initially_zero(self) -> None:
        """初期状態で sample_count が 0。"""
        enroller = PassphraseEnroller(min_unique_phonemes=None)
        assert enroller.sample_count == 0

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    @patch("voice_auth_engine.speaker_verifier.validate_passphrase")
    @patch("voice_auth_engine.speaker_verifier.transcribe")
    def test_phoneme_check_disabled(
        self,
        mock_transcribe: MagicMock,
        mock_validate: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """min_unique_phonemes=None で音素チェックスキップ。"""
        mock_extract.return_value = make_embedding([1.0, 0.0, 0.0])
        enroller = PassphraseEnroller(min_speech_seconds=0.1, min_unique_phonemes=None)
        enroller.add_sample(make_audio(1.0))
        mock_transcribe.assert_not_called()
        mock_validate.assert_not_called()


class TestPassphraseVerifier:
    """PassphraseVerifier のテスト。"""

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    def test_verify_accepted(self, mock_extract: MagicMock) -> None:
        """類似度 ≥ threshold で accepted=True。"""
        enrolled = make_embedding([1.0, 0.0, 0.0])
        mock_extract.return_value = make_embedding([1.0, 0.0, 0.0])
        verifier = PassphraseVerifier(
            enrolled,
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=None,
        )
        result = verifier.verify(make_audio(1.0))
        assert result.accepted is True
        assert result.score == pytest.approx(1.0)

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    def test_verify_rejected(self, mock_extract: MagicMock) -> None:
        """類似度 < threshold で accepted=False。"""
        enrolled = make_embedding([1.0, 0.0, 0.0])
        mock_extract.return_value = make_embedding([0.0, 1.0, 0.0])
        verifier = PassphraseVerifier(
            enrolled,
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=None,
        )
        result = verifier.verify(make_audio(1.0))
        assert result.accepted is False
        assert result.score == pytest.approx(0.0)

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    def test_verify_returns_score(self, mock_extract: MagicMock) -> None:
        """スコアが [-1.0, 1.0] の範囲。"""
        enrolled = make_embedding([1.0, 0.0, 0.0])
        mock_extract.return_value = make_embedding([-1.0, 0.0, 0.0])
        verifier = PassphraseVerifier(
            enrolled,
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=None,
        )
        result = verifier.verify(make_audio(1.0))
        assert -1.0 <= result.score <= 1.0

    def test_threshold_property(self) -> None:
        """プロパティの値が初期化時と一致。"""
        verifier = PassphraseVerifier(
            make_embedding([1.0, 0.0, 0.0]),
            threshold=0.7,
            min_unique_phonemes=None,
        )
        assert verifier.threshold == 0.7

    @patch("voice_auth_engine.speaker_verifier.extract_embedding")
    @patch("voice_auth_engine.speaker_verifier.validate_passphrase")
    @patch("voice_auth_engine.speaker_verifier.transcribe")
    def test_phoneme_check_disabled(
        self,
        mock_transcribe: MagicMock,
        mock_validate: MagicMock,
        mock_extract: MagicMock,
    ) -> None:
        """min_unique_phonemes=None で音素チェックスキップ。"""
        enrolled = make_embedding([1.0, 0.0, 0.0])
        mock_extract.return_value = make_embedding([1.0, 0.0, 0.0])
        verifier = PassphraseVerifier(
            enrolled,
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=None,
        )
        verifier.verify(make_audio(1.0))
        mock_transcribe.assert_not_called()
        mock_validate.assert_not_called()
