"""voice_auth モジュールの統合テスト（実音声）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_auth_engine.audio_validator import EmptyAudioError, InsufficientDurationError
from voice_auth_engine.passphrase_validator import PhonemeConsistencyError
from voice_auth_engine.voice_auth import Passphrase, VoiceAuth

from .audio_factory import generate_silence_samples, make_audio_data

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def auth() -> VoiceAuth:
    """統合テスト用 VoiceAuth。"""
    return VoiceAuth(threshold=0.8)


class TestVoiceAuthIntegration:
    """実音声を使ったパスフレーズ認証の統合テスト。"""

    def test_enroll_and_verify_same_speaker(self, auth: VoiceAuth) -> None:
        """登録→本人認証で accepted=True。"""
        passphrases = [auth.extract_passphrase(FIXTURES_DIR / "speaker_a_enroll.mp3")]
        selected, _ = auth.select_passphrase(passphrases)

        passphrase = auth.extract_passphrase(FIXTURES_DIR / "speaker_a_verify.mp3")
        result = auth.verify_passphrase(passphrase, selected)
        assert result.voiceprint_accepted is True
        assert result.voiceprint_score > auth.threshold

    @pytest.mark.parametrize(
        "other_speaker",
        ["speaker_b_verify.mp3", "speaker_c_verify.mp3", "speaker_d_verify.mp3"],
    )
    def test_reject_different_speaker(self, auth: VoiceAuth, other_speaker: str) -> None:
        """登録→他人認証で accepted=False。"""
        passphrases = [auth.extract_passphrase(FIXTURES_DIR / "speaker_a_enroll.mp3")]
        selected, _ = auth.select_passphrase(passphrases)

        passphrase = auth.extract_passphrase(FIXTURES_DIR / other_speaker)
        result = auth.verify_passphrase(passphrase, selected)
        assert result.voiceprint_accepted is False
        assert result.voiceprint_score < auth.threshold

    @pytest.mark.parametrize(
        "other_speaker",
        ["speaker_b_verify.mp3", "speaker_c_verify.mp3", "speaker_d_verify.mp3"],
    )
    def test_same_speaker_score_higher_than_different_speaker(
        self, auth: VoiceAuth, other_speaker: str
    ) -> None:
        """本人のスコアが他人のスコアより高い。"""
        passphrases = [auth.extract_passphrase(FIXTURES_DIR / "speaker_a_enroll.mp3")]
        selected, _ = auth.select_passphrase(passphrases)

        same_passphrase = auth.extract_passphrase(FIXTURES_DIR / "speaker_a_verify.mp3")
        diff_passphrase = auth.extract_passphrase(FIXTURES_DIR / other_speaker)
        same_result = auth.verify_passphrase(same_passphrase, selected)
        diff_result = auth.verify_passphrase(diff_passphrase, selected)
        assert same_result.voiceprint_score > diff_result.voiceprint_score

    def test_embedding_serialization_roundtrip(self, auth: VoiceAuth) -> None:
        """埋め込みベクトルのシリアライズ→デシリアライズ後も認証可能。"""
        from voice_auth_engine.embedding_extractor import Embedding

        passphrases = [auth.extract_passphrase(FIXTURES_DIR / "speaker_a_enroll.mp3")]
        selected, _ = auth.select_passphrase(passphrases)

        restored = Embedding.from_bytes(selected.embedding.to_bytes())
        restored_enrolled = Passphrase(
            embedding=restored,
            phoneme=selected.phoneme,
            transcription=selected.transcription,
            speech_duration=selected.speech_duration,
        )
        passphrase = auth.extract_passphrase(FIXTURES_DIR / "speaker_a_verify.mp3")
        result = auth.verify_passphrase(passphrase, restored_enrolled)
        assert result.voiceprint_accepted is True

    def test_silence_raises_empty_audio_error(self, auth: VoiceAuth) -> None:
        """無音音声で EmptyAudioError が発生する。"""
        silence = make_audio_data(generate_silence_samples(duration=5.0))
        with pytest.raises(EmptyAudioError):
            auth.extract_passphrase(silence)

    def test_short_speech_raises_insufficient_duration_error(
        self,
        auth: VoiceAuth,
    ) -> None:
        """発話時間が短い音声で InsufficientDurationError が発生する。"""
        with pytest.raises(InsufficientDurationError):
            auth.extract_passphrase(FIXTURES_DIR / "digits_clear.mp3")


@pytest.fixture
def phoneme_auth() -> VoiceAuth:
    """音素照合を有効にした統合テスト用 VoiceAuth。"""
    return VoiceAuth(threshold=0.8, phoneme_threshold=0.3)


class TestPhonemeVerificationIntegration:
    """音素ベースのパスフレーズ照合の統合テスト。"""

    def test_select_with_phoneme_threshold(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """phoneme_threshold 有効で選択すると phonemes を含む Passphrase が返る。"""
        passphrases = [
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3"),
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3"),
        ]
        selected, index = phoneme_auth.select_passphrase(passphrases)

        assert len(selected.phoneme.values) > 0
        assert selected is passphrases[index]

    def test_select_three_samples(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """3サンプルで medoid が選択される。"""
        passphrases = [
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3"),
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3"),
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_3.mp3"),
        ]
        selected, index = phoneme_auth.select_passphrase(passphrases)

        assert len(selected.phoneme.values) > 0
        assert 0 <= index <= 2

    def test_same_speaker_same_passphrase_accepted(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """本人+同一パスフレーズで accepted=True, passphrase_accepted=True。"""
        passphrases = [
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3"),
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3"),
        ]
        selected, _ = phoneme_auth.select_passphrase(passphrases)

        passphrase = phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_verify.mp3")
        result = phoneme_auth.verify_passphrase(passphrase, selected)

        assert result.voiceprint_accepted is True
        assert result.passphrase_accepted is True
        assert result.passphrase_score is not None
        assert phoneme_auth._phoneme_threshold is not None
        assert result.passphrase_score <= phoneme_auth._phoneme_threshold

    def test_same_speaker_different_passphrase_rejected(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """本人+異なるパスフレーズで accepted=False（話者OK, 音素NG）。"""
        passphrases = [
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3"),
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3"),
        ]
        selected, _ = phoneme_auth.select_passphrase(passphrases)

        passphrase = phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_b.mp3")
        result = phoneme_auth.verify_passphrase(passphrase, selected)

        assert result.voiceprint_accepted is False
        assert result.voiceprint_score >= phoneme_auth.threshold  # 話者は本人
        assert result.passphrase_accepted is False  # 音素が不一致

    def test_different_speaker_same_passphrase_rejected(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """他人+同一パスフレーズで accepted=False（話者NG, 音素OK）。"""
        passphrases = [
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3"),
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3"),
        ]
        selected, _ = phoneme_auth.select_passphrase(passphrases)

        passphrase = phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_b_phrase_a.mp3")
        result = phoneme_auth.verify_passphrase(passphrase, selected)

        assert result.voiceprint_accepted is False
        assert result.voiceprint_score < phoneme_auth.threshold  # 話者が異なる
        assert result.passphrase_accepted is True  # 音素は一致

    def test_inconsistent_passphrases_raises_error(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """異なるパスフレーズで登録すると PhonemeConsistencyError。"""
        passphrases = [
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3"),
            phoneme_auth.extract_passphrase(FIXTURES_DIR / "speaker_a_phrase_b.mp3"),
        ]
        with pytest.raises(PhonemeConsistencyError):
            phoneme_auth.select_passphrase(passphrases)
