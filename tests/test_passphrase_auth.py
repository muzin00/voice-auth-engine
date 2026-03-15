"""passphrase_auth モジュールのテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.audio_validator import EmptyAudioError
from voice_auth_engine.embedding_extractor import Embedding
from voice_auth_engine.passphrase_auth import (
    Passphrase,
    PassphraseAuth,
)
from voice_auth_engine.passphrase_validator import PhonemeConsistencyError
from voice_auth_engine.phoneme_extractor import Phoneme

from .audio_factory import make_audio, make_embedding, make_segments


@pytest.fixture
def auth() -> PassphraseAuth:
    """テスト用 PassphraseAuth。"""
    return PassphraseAuth(
        threshold=0.5,
        min_speech_seconds=0.1,
        min_unique_phonemes=None,
    )


class TestSelectPassphrase:
    """PassphraseAuth.select_passphrase のテスト。"""

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_single_sample(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """1サンプルでそのまま返る。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        passphrases = [auth.extract_passphrase(audio)]
        selected, index = auth.select_passphrase(passphrases)
        assert index == 0
        assert selected is passphrases[0]
        np.testing.assert_array_equal(selected.embedding.values, [1.0, 0.0, 0.0])

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_multiple_samples_selects_medoid(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """複数サンプルで medoid が選択される。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        # サンプル0,1は近く、サンプル2は遠い → medoid はサンプル0またはサンプル1
        mock_extract_emb.side_effect = [
            make_embedding([1.0, 0.0, 0.0]),
            make_embedding([0.9, 0.1, 0.0]),
            make_embedding([0.0, 0.0, 1.0]),
        ]

        passphrases = [auth.extract_passphrase(audio) for _ in range(3)]
        selected, index = auth.select_passphrase(passphrases)
        assert index in (0, 1)
        assert selected is passphrases[index]

    def test_empty_raises(self, auth: PassphraseAuth) -> None:
        """空リストで ValueError。"""
        with pytest.raises(ValueError, match="passphrases が空です"):
            auth.select_passphrase([])

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    def test_extract_no_speech_raises(
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

        with pytest.raises(EmptyAudioError):
            auth.extract_passphrase(audio)

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_returns_passphrase_and_index(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """select_passphrase() が tuple[Passphrase, int] を返す。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        passphrases = [auth.extract_passphrase(audio)]
        selected, index = auth.select_passphrase(passphrases)
        assert isinstance(selected, Passphrase)
        assert isinstance(index, int)


class TestVerifyPassphrase:
    """PassphraseAuth.verify_passphrase のテスト。"""

    def test_verify_accepted(self, auth: PassphraseAuth) -> None:
        """類似度 >= threshold で accepted=True。"""
        enrolled = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=[]),
            transcription="",
            speech_duration=1.0,
        )
        passphrase = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=[]),
            transcription="",
            speech_duration=1.0,
        )

        result = auth.verify_passphrase(passphrase, enrolled)
        assert result.voiceprint_accepted is True
        assert result.voiceprint_score == pytest.approx(1.0)

    def test_verify_rejected(self, auth: PassphraseAuth) -> None:
        """類似度 < threshold で accepted=False。"""
        enrolled = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=[]),
            transcription="",
            speech_duration=1.0,
        )
        passphrase = Passphrase(
            embedding=make_embedding([0.0, 1.0, 0.0]),
            phoneme=Phoneme(values=[]),
            transcription="",
            speech_duration=1.0,
        )

        result = auth.verify_passphrase(passphrase, enrolled)
        assert result.voiceprint_accepted is False
        assert result.voiceprint_score == pytest.approx(0.0)


class TestSelectPassphrasePhonemes:
    """select_passphrase の音素整合性チェックのテスト。"""

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
        mock_extract_phonemes: MagicMock,
        *,
        phoneme_lists: list[list[str]],
        embedding_lists: list[list[float]] | None = None,
    ) -> AudioData:
        """共通モック設定。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        if embedding_lists is not None:
            mock_extract_emb.side_effect = [make_embedding(e) for e in embedding_lists]
        else:
            mock_extract_emb.side_effect = [make_embedding([1.0, 0.0, 0.0]) for _ in phoneme_lists]
        mock_transcribe.side_effect = [
            MagicMock(text=f"text{i}") for i in range(len(phoneme_lists))
        ]
        mock_extract_phonemes.side_effect = [Phoneme(values=p) for p in phoneme_lists]
        return audio

    @patch("voice_auth_engine.passphrase_auth.extract_phonemes")
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
        mock_extract_phonemes: MagicMock,
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
            mock_extract_phonemes,
            phoneme_lists=[phonemes],
        )
        result = auth_with_phonemes.extract_passphrase(audio)
        assert result.phoneme.values == phonemes
        assert isinstance(result.embedding, Embedding)

    @patch("voice_auth_engine.passphrase_auth.extract_phonemes")
    @patch("voice_auth_engine.passphrase_auth.transcribe")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.load_audio")
    def test_selects_medoid_by_embedding(
        self,
        mock_load: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_extract_emb: MagicMock,
        mock_transcribe: MagicMock,
        mock_extract_phonemes: MagicMock,
        auth_with_phonemes: PassphraseAuth,
    ) -> None:
        """select_passphrase() が embedding の medoid でサンプルを選択する。"""
        # サンプル0,1は近い embedding、サンプル2は遠い
        # → medoid はサンプル0またはサンプル1
        phoneme_lists = [
            ["a", "i", "u", "e", "o"],
            ["a", "i", "u", "e", "o"],
            ["a", "i", "u", "e", "o"],
        ]
        embedding_lists = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ]
        audio = self._setup_mocks(
            mock_load,
            mock_detect,
            mock_extract_sp,
            mock_extract_emb,
            mock_transcribe,
            mock_extract_phonemes,
            phoneme_lists=phoneme_lists,
            embedding_lists=embedding_lists,
        )
        passphrases = [auth_with_phonemes.extract_passphrase(audio) for _ in range(3)]
        selected, index = auth_with_phonemes.select_passphrase(passphrases)
        assert index in (0, 1)
        assert selected is passphrases[index]

    @patch("voice_auth_engine.passphrase_auth.extract_phonemes")
    @patch("voice_auth_engine.passphrase_auth.transcribe")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.load_audio")
    def test_raises_on_inconsistent_phonemes(
        self,
        mock_load: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_extract_emb: MagicMock,
        mock_transcribe: MagicMock,
        mock_extract_phonemes: MagicMock,
    ) -> None:
        """音素列の距離が閾値を超えると PhonemeConsistencyError。"""
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
            mock_extract_phonemes,
            phoneme_lists=phoneme_lists,
        )
        passphrases = [auth.extract_passphrase(audio) for _ in range(2)]
        with pytest.raises(PhonemeConsistencyError, match="音素列の不整合"):
            auth.select_passphrase(passphrases)

    @patch("voice_auth_engine.passphrase_auth.load_audio")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    def test_without_phoneme_threshold(
        self,
        mock_extract_emb: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_load: MagicMock,
        auth: PassphraseAuth,
    ) -> None:
        """phoneme_threshold=None でも正常に選択できる。"""
        audio = make_audio(1.0)
        mock_load.return_value = audio
        mock_detect.return_value = make_segments(audio)
        mock_extract_sp.return_value = audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])

        passphrases = [auth.extract_passphrase(audio)]
        selected, index = auth.select_passphrase(passphrases)
        assert index == 0
        assert selected is passphrases[0]


class TestVerifyPassphrasePhonemes:
    """verify_passphrase の音素照合ゲート付きテスト。"""

    @pytest.fixture
    def auth_with_phoneme_gate(self) -> PassphraseAuth:
        """phoneme_threshold 有効の PassphraseAuth。"""
        return PassphraseAuth(
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=None,
            phoneme_threshold=0.3,
        )

    def test_phoneme_match_accepted(
        self,
        auth_with_phoneme_gate: PassphraseAuth,
    ) -> None:
        """話者一致 + 音素一致で accepted=True。"""
        enrolled = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=["a", "i", "u", "e", "o"]),
            transcription="test",
            speech_duration=1.0,
        )
        passphrase = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=["a", "i", "u", "e", "o"]),
            transcription="test",
            speech_duration=1.0,
        )

        result = auth_with_phoneme_gate.verify_passphrase(passphrase, enrolled)

        assert result.voiceprint_accepted is True
        assert result.passphrase_accepted is True
        assert result.passphrase_score == pytest.approx(0.0)

    def test_phoneme_mismatch_rejected(
        self,
        auth_with_phoneme_gate: PassphraseAuth,
    ) -> None:
        """話者一致 + 音素不一致で accepted=False。"""
        enrolled = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=["a", "i", "u", "e", "o"]),
            transcription="test",
            speech_duration=1.0,
        )
        passphrase = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=["k", "a", "k", "i", "k"]),
            transcription="test",
            speech_duration=1.0,
        )

        result = auth_with_phoneme_gate.verify_passphrase(passphrase, enrolled)

        assert result.voiceprint_accepted is False
        assert result.passphrase_accepted is False
        assert result.passphrase_score is not None
        assert result.passphrase_score > 0.3

    def test_speaker_mismatch_with_phoneme_match_rejected(
        self,
        auth_with_phoneme_gate: PassphraseAuth,
    ) -> None:
        """話者不一致 + 音素一致でも accepted=False。"""
        enrolled = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=["a", "i", "u", "e", "o"]),
            transcription="test",
            speech_duration=1.0,
        )
        passphrase = Passphrase(
            embedding=make_embedding([0.0, 1.0, 0.0]),
            phoneme=Phoneme(values=["a", "i", "u", "e", "o"]),
            transcription="test",
            speech_duration=1.0,
        )

        result = auth_with_phoneme_gate.verify_passphrase(passphrase, enrolled)

        assert result.voiceprint_accepted is False
        assert result.passphrase_accepted is True
        assert result.voiceprint_score == pytest.approx(0.0)

    def test_no_phoneme_threshold_skips_passphrase_check(self, auth: PassphraseAuth) -> None:
        """phoneme_threshold=None で音素照合無効。"""
        enrolled = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=[]),
            transcription="",
            speech_duration=1.0,
        )
        passphrase = Passphrase(
            embedding=make_embedding([1.0, 0.0, 0.0]),
            phoneme=Phoneme(values=[]),
            transcription="",
            speech_duration=1.0,
        )

        result = auth.verify_passphrase(passphrase, enrolled)

        assert result.voiceprint_accepted is True
        assert result.passphrase_score is None
        assert result.passphrase_accepted is None


class TestExtractPassphrase:
    """extract_passphrase の戻り値テスト。"""

    @patch("voice_auth_engine.passphrase_auth.extract_phonemes")
    @patch("voice_auth_engine.passphrase_auth.transcribe")
    @patch("voice_auth_engine.passphrase_auth.extract_embedding")
    @patch("voice_auth_engine.passphrase_auth.extract_speech")
    @patch("voice_auth_engine.passphrase_auth.detect_speech")
    @patch("voice_auth_engine.passphrase_auth.load_audio")
    def test_result_contains_all_fields(
        self,
        mock_load: MagicMock,
        mock_detect: MagicMock,
        mock_extract_sp: MagicMock,
        mock_extract_emb: MagicMock,
        mock_transcribe: MagicMock,
        mock_extract_phonemes: MagicMock,
    ) -> None:
        """全フィールド（embedding, phoneme, transcription, speech_duration）が返る。"""
        auth = PassphraseAuth(
            threshold=0.5,
            min_speech_seconds=0.1,
            min_unique_phonemes=5,
        )
        original_audio = make_audio(5.0)
        speech_audio = make_audio(1.5)
        mock_load.return_value = original_audio
        mock_detect.return_value = make_segments(original_audio)
        mock_extract_sp.return_value = speech_audio
        mock_extract_emb.return_value = make_embedding([1.0, 0.0, 0.0])
        mock_transcribe.return_value = MagicMock(text="こんにちは世界")
        phonemes = ["k", "o", "N", "n", "i", "ch", "w", "a", "s", "e", "k", "a", "i"]
        mock_extract_phonemes.return_value = Phoneme(values=phonemes)

        result = auth.extract_passphrase(original_audio)

        assert isinstance(result, Passphrase)
        assert isinstance(result.embedding, Embedding)
        assert result.phoneme.values == phonemes
        assert result.transcription == "こんにちは世界"
        assert result.speech_duration == pytest.approx(1.5)
