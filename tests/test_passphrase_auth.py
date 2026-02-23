"""passphrase_auth モジュールのテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.audio_validator import EmptyAudioError
from voice_auth_engine.embedding_extractor import Embedding
from voice_auth_engine.passphrase_auth import (
    EnrollmentResult,
    PassphraseAuth,
    PassphraseAuthEnroller,
    PassphraseAuthVerifier,
    PassphraseEnrollmentError,
)
from voice_auth_engine.passphrase_validator import PassphraseInfo

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
        result = enroller.enroll()
        np.testing.assert_array_equal(result.embedding.values, [1.0, 0.0, 0.0])

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
        result = enroller.enroll()
        np.testing.assert_array_almost_equal(result.embedding.values, [0.5, 0.5, 0.0])

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
        result = enroller.enroll()
        assert isinstance(result, EnrollmentResult)
        assert isinstance(result.embedding, Embedding)

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


class TestPassphraseAuthPhonemes:
    """音素収集・メドイド選択・整合性チェックのテスト。"""

    @pytest.fixture
    def auth_with_phonemes(self) -> PassphraseAuth:
        """phoneme_threshold 有効の PassphraseAuth。"""
        return PassphraseAuth(
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=None,
            phoneme_threshold=0.3,
        )

    def _setup_mocks(
        self,
        mock_load: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_extract_emb: MagicMock,
        mock_transcribe: MagicMock,
        mock_analyze: MagicMock,
        *,
        phoneme_lists: list[list[str]],
    ) -> AudioData:
        """共通モック設定。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.side_effect = [make_embedding([1.0, 0.0, 0.0]) for _ in phoneme_lists]
        mock_transcribe.side_effect = [
            MagicMock(text=f"text{i}") for i in range(len(phoneme_lists))
        ]
        mock_analyze.side_effect = [
            PassphraseInfo(
                text=f"text{i}",
                phonemes=p,
                unique_phonemes=set(p),
                unique_count=len(set(p)),
            )
            for i, p in enumerate(phoneme_lists)
        ]
        return audio

    @patch("voice_auth_engine.passphrase_auth.analyze_passphrase")
    @patch("voice_auth_engine.passphrase_auth.transcribe")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.load_audio")
    def test_extract_passphrase_returns_phonemes(
        self,
        mock_load: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_extract_emb: MagicMock,
        mock_transcribe: MagicMock,
        mock_analyze: MagicMock,
        auth_with_phonemes: PassphraseAuth,
    ) -> None:
        """extract_passphrase が音素列を含む結果を返す。"""
        phonemes = ["a", "i", "u", "e", "o"]
        audio = self._setup_mocks(
            mock_load,
            mock_detect,
            mock_extract_sp,
            mock_extract_emb,
            mock_transcribe,
            mock_analyze,
            phoneme_lists=[phonemes],
        )
        result = auth_with_phonemes.extract_passphrase(audio)
        assert result.phonemes == phonemes
        assert isinstance(result.embedding, Embedding)

    @patch("voice_auth_engine.passphrase_auth.analyze_passphrase")
    @patch("voice_auth_engine.passphrase_auth.transcribe")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.load_audio")
    def test_enroll_selects_medoid_phonemes(
        self,
        mock_load: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_extract_emb: MagicMock,
        mock_transcribe: MagicMock,
        mock_analyze: MagicMock,
        auth_with_phonemes: PassphraseAuth,
    ) -> None:
        """enroll() がメドイドで選ばれた基準音素列を返す。"""
        # サンプル0,1は同一、サンプル2は1箇所異なる
        # d(0,1)=0.0, d(0,2)=0.2, d(1,2)=0.2
        # メドイド: サンプル0（距離合計0.2、同率のサンプル1より先頭）
        phoneme_lists = [
            ["a", "i", "u", "e", "o"],
            ["a", "i", "u", "e", "o"],
            ["a", "i", "u", "e", "a"],
        ]
        audio = self._setup_mocks(
            mock_load,
            mock_detect,
            mock_extract_sp,
            mock_extract_emb,
            mock_transcribe,
            mock_analyze,
            phoneme_lists=phoneme_lists,
        )
        enroller = auth_with_phonemes.create_enroller()
        for _ in range(3):
            enroller.add_sample(audio)
        result = enroller.enroll()
        assert isinstance(result, EnrollmentResult)
        assert result.phonemes == ["a", "i", "u", "e", "o"]

    @patch("voice_auth_engine.passphrase_auth.analyze_passphrase")
    @patch("voice_auth_engine.passphrase_auth.transcribe")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.load_audio")
    def test_enroll_raises_on_inconsistent_phonemes(
        self,
        mock_load: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_extract_emb: MagicMock,
        mock_transcribe: MagicMock,
        mock_analyze: MagicMock,
    ) -> None:
        """音素列の距離が閾値を超えると PassphraseEnrollmentError。"""
        auth = PassphraseAuth(
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=None,
            phoneme_threshold=0.1,  # 厳しい閾値
        )
        # d(0,1) = 0.4 > 0.1 → エラー
        phoneme_lists = [
            ["a", "i", "u", "e", "o"],
            ["k", "a", "u", "e", "o"],
        ]
        audio = self._setup_mocks(
            mock_load,
            mock_detect,
            mock_extract_sp,
            mock_extract_emb,
            mock_transcribe,
            mock_analyze,
            phoneme_lists=phoneme_lists,
        )
        enroller = auth.create_enroller()
        enroller.add_sample(audio)
        enroller.add_sample(audio)
        with pytest.raises(PassphraseEnrollmentError, match="音素列の不整合"):
            enroller.enroll()

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_enroll_without_phoneme_threshold_returns_empty_phonemes(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """phoneme_threshold=None の場合、空の音素列を返す。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        enroller = auth.create_enroller()
        enroller.add_sample(audio)
        result = enroller.enroll()
        assert result.phonemes == []
