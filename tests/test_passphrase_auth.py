"""passphrase_auth モジュールのテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.audio_validator import EmptyAudioError
from voice_auth_engine.embedding_extractor import Embedding
from voice_auth_engine.passphrase_auth import (
    PassphraseAuth,
    PassphraseAuthEnroller,
    PassphraseAuthVerifier,
)

from .audio_factory import make_audio, make_embedding, make_segments


@pytest.fixture
def auth() -> PassphraseAuth:
    """テスト用 PassphraseAuth。"""
    return PassphraseAuth(
        threshold=0.5,
        min_speech_seconds=0.1,
        min_unique_phonemes=None,
    )


class TestPassphraseAuthEnroller:
    """PassphraseAuthEnroller のテスト。"""

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_add_sample_increments_count(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """add_sample 後に sample_count が増加する。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        enroller = auth.create_enroller()
        assert enroller.sample_count == 0
        enroller.add_sample(audio)
        assert enroller.sample_count == 1
        enroller.add_sample(audio)
        assert enroller.sample_count == 2

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_enroll_single_sample(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """1サンプルで正常に登録できる。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        enroller = auth.create_enroller()
        enroller.add_sample(audio)
        embedding = enroller.enroll()
        np.testing.assert_array_equal(embedding.values, [1.0, 0.0, 0.0])

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_enroll_multiple_samples(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """複数サンプルで平均ベクトルが返る。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.side_effect = [
            make_embedding([1.0, 0.0, 0.0]),
            make_embedding([0.0, 1.0, 0.0]),
        ]

        enroller = auth.create_enroller()
        enroller.add_sample(audio)
        enroller.add_sample(audio)
        embedding = enroller.enroll()
        np.testing.assert_array_almost_equal(embedding.values, [0.5, 0.5, 0.0])

    def test_enroll_no_samples_raises(self, auth: PassphraseAuth) -> None:
        """サンプル未蓄積で ValueError。"""
        enroller = auth.create_enroller()
        with pytest.raises(ValueError, match="サンプルが蓄積されていません"):
            enroller.enroll()

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    def test_add_sample_no_speech_raises(
        self,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """VAD で音声なしの場合 EmptyAudioError。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio, empty=True)
        mock_extract_sp.return_value = AudioData(
            samples=np.array([], dtype=np.int16), sample_rate=16000
        )

        enroller = auth.create_enroller()
        with pytest.raises(EmptyAudioError):
            enroller.add_sample(audio)

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_enroll_returns_embedding(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """enroll() が Embedding を返す。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        enroller = auth.create_enroller()
        enroller.add_sample(audio)
        embedding = enroller.enroll()
        assert isinstance(embedding, Embedding)

    def test_create_enroller_returns_correct_type(self, auth: PassphraseAuth) -> None:
        """create_enroller が PassphraseAuthEnroller を返す。"""
        enroller = auth.create_enroller()
        assert isinstance(enroller, PassphraseAuthEnroller)


class TestPassphraseAuthVerifier:
    """PassphraseAuthVerifier のテスト。"""

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_verify_accepted(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """類似度 >= threshold で accepted=True。"""
        audio = make_audio(1.0)
        enrolled = make_embedding([1.0, 0.0, 0.0])
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        verifier = auth.create_verifier(enrolled)
        result = verifier.verify(audio)
        assert result.accepted is True
        assert result.score == pytest.approx(1.0)

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_verify_rejected(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """類似度 < threshold で accepted=False。"""
        audio = make_audio(1.0)
        enrolled = make_embedding([1.0, 0.0, 0.0])
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([0.0, 1.0, 0.0])

        verifier = auth.create_verifier(enrolled)
        result = verifier.verify(audio)
        assert result.accepted is False
        assert result.score == pytest.approx(0.0)

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    def test_verify_no_speech_raises(
        self,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """VAD で音声なしの場合 EmptyAudioError。"""
        audio = make_audio(1.0)
        enrolled = make_embedding([1.0, 0.0, 0.0])
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio, empty=True)
        mock_extract_sp.return_value = AudioData(
            samples=np.array([], dtype=np.int16), sample_rate=16000
        )

        verifier = auth.create_verifier(enrolled)
        with pytest.raises(EmptyAudioError):
            verifier.verify(audio)

    def test_create_verifier_returns_correct_type(self, auth: PassphraseAuth) -> None:
        """create_verifier が PassphraseAuthVerifier を返す。"""
        enrolled = make_embedding([1.0, 0.0, 0.0])
        verifier = auth.create_verifier(enrolled)
        assert isinstance(verifier, PassphraseAuthVerifier)
