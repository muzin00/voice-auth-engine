"""passphrase_auth モジュールの統合テスト（実音声）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_auth_engine.audio_validator import EmptyAudioError, InsufficientDurationError
from voice_auth_engine.passphrase_auth import PassphraseAuth, PassphraseEnrollmentError

from .audio_factory import generate_silence_samples, make_audio_data
from .conftest import requires_campplus_model, requires_sense_voice_model, requires_silero_vad_model

FIXTURES_DIR = Path(__file__).parent / "fixtures"

requires_all_models = [
    requires_silero_vad_model,
    requires_sense_voice_model,
    requires_campplus_model,
]


@pytest.fixture
def auth() -> PassphraseAuth:
    """統合テスト用 PassphraseAuth。"""
    return PassphraseAuth(threshold=0.8)


@pytest.mark.parametrize("_marker", [pytest.param(None, marks=requires_all_models)])
class TestPassphraseAuthIntegration:
    """実音声を使ったパスフレーズ認証の統合テスト。"""

    def test_enroll_and_verify_same_speaker(self, auth: PassphraseAuth, _marker: None) -> None:
        """登録→本人認証で accepted=True。"""
        enroller = auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_enroll.mp3")
        result = enroller.enroll()

        verifier = auth.create_verifier(result.embedding)
        result = verifier.verify(FIXTURES_DIR / "speaker_a_verify.mp3")
        assert result.accepted is True
        assert result.score > auth.threshold

    @pytest.mark.parametrize(
        "other_speaker",
        ["speaker_b_verify.mp3", "speaker_c_verify.mp3", "speaker_d_verify.mp3"],
    )
    def test_reject_different_speaker(
        self, auth: PassphraseAuth, _marker: None, other_speaker: str
    ) -> None:
        """登録→他人認証で accepted=False。"""
        enroller = auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_enroll.mp3")
        result = enroller.enroll()

        verifier = auth.create_verifier(result.embedding)
        result = verifier.verify(FIXTURES_DIR / other_speaker)
        assert result.accepted is False
        assert result.score < auth.threshold

    @pytest.mark.parametrize(
        "other_speaker",
        ["speaker_b_verify.mp3", "speaker_c_verify.mp3", "speaker_d_verify.mp3"],
    )
    def test_same_speaker_score_higher_than_different_speaker(
        self, auth: PassphraseAuth, _marker: None, other_speaker: str
    ) -> None:
        """本人のスコアが他人のスコアより高い。"""
        enroller = auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_enroll.mp3")
        result = enroller.enroll()

        verifier = auth.create_verifier(result.embedding)
        same_result = verifier.verify(FIXTURES_DIR / "speaker_a_verify.mp3")
        diff_result = verifier.verify(FIXTURES_DIR / other_speaker)
        assert same_result.score > diff_result.score

    def test_embedding_serialization_roundtrip(self, auth: PassphraseAuth, _marker: None) -> None:
        """埋め込みベクトルのシリアライズ→デシリアライズ後も認証可能。"""
        from voice_auth_engine.embedding_extractor import Embedding

        enroller = auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_enroll.mp3")
        enrollment = enroller.enroll()

        restored = Embedding.from_bytes(enrollment.embedding.to_bytes())
        verifier = auth.create_verifier(restored)
        result = verifier.verify(FIXTURES_DIR / "speaker_a_verify.mp3")
        assert result.accepted is True

    def test_silence_raises_empty_audio_error(self, auth: PassphraseAuth, _marker: None) -> None:
        """無音音声で EmptyAudioError が発生する。"""
        silence = make_audio_data(generate_silence_samples(duration=5.0))
        enroller = auth.create_enroller()
        with pytest.raises(EmptyAudioError):
            enroller.add_sample(silence)

    def test_short_speech_raises_insufficient_duration_error(
        self, auth: PassphraseAuth, _marker: None
    ) -> None:
        """発話時間が短い音声で InsufficientDurationError が発生する。"""
        enroller = auth.create_enroller()
        with pytest.raises(InsufficientDurationError):
            enroller.add_sample(FIXTURES_DIR / "digits_clear.mp3")


@pytest.fixture
def phoneme_auth() -> PassphraseAuth:
    """音素照合を有効にした統合テスト用 PassphraseAuth。"""
    return PassphraseAuth(threshold=0.8, phoneme_threshold=0.3)


@pytest.mark.parametrize("_marker", [pytest.param(None, marks=requires_all_models)])
class TestPhonemeVerificationIntegration:
    """音素ベースのパスフレーズ照合の統合テスト。"""

    def test_enroll_with_phoneme_threshold(
        self, phoneme_auth: PassphraseAuth, _marker: None
    ) -> None:
        """phoneme_threshold 有効で登録すると phonemes が返る。"""
        enroller = phoneme_auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")
        result = enroller.enroll()

        assert len(result.phonemes) > 0

    def test_enroll_three_samples_selects_medoid(
        self, phoneme_auth: PassphraseAuth, _marker: None
    ) -> None:
        """3サンプル登録でメドイドが選択され phonemes が返る。"""
        enroller = phoneme_auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_3.mp3")
        result = enroller.enroll()

        assert enroller.sample_count == 3
        assert len(result.phonemes) > 0

    def test_same_speaker_same_passphrase_accepted(
        self, phoneme_auth: PassphraseAuth, _marker: None
    ) -> None:
        """本人+同一パスフレーズで accepted=True, passphrase_accepted=True。"""
        enroller = phoneme_auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")
        enrollment = enroller.enroll()

        verifier = phoneme_auth.create_verifier(enrollment.embedding, enrollment.phonemes)
        result = verifier.verify(FIXTURES_DIR / "speaker_a_phrase_a_verify.mp3")

        assert result.accepted is True
        assert result.passphrase_accepted is True
        assert result.phoneme_score is not None
        assert phoneme_auth._phoneme_threshold is not None
        assert result.phoneme_score <= phoneme_auth._phoneme_threshold

    def test_same_speaker_different_passphrase_rejected(
        self, phoneme_auth: PassphraseAuth, _marker: None
    ) -> None:
        """本人+異なるパスフレーズで accepted=False（話者OK, 音素NG）。"""
        enroller = phoneme_auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")
        enrollment = enroller.enroll()

        verifier = phoneme_auth.create_verifier(enrollment.embedding, enrollment.phonemes)
        result = verifier.verify(FIXTURES_DIR / "speaker_a_phrase_b.mp3")

        assert result.accepted is False
        assert result.score >= phoneme_auth.threshold  # 話者は本人
        assert result.passphrase_accepted is False  # 音素が不一致

    def test_different_speaker_same_passphrase_rejected(
        self, phoneme_auth: PassphraseAuth, _marker: None
    ) -> None:
        """他人+同一パスフレーズで accepted=False（話者NG, 音素OK）。"""
        enroller = phoneme_auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")
        enrollment = enroller.enroll()

        verifier = phoneme_auth.create_verifier(enrollment.embedding, enrollment.phonemes)
        result = verifier.verify(FIXTURES_DIR / "speaker_b_phrase_a.mp3")

        assert result.accepted is False
        assert result.score < phoneme_auth.threshold  # 話者が異なる
        assert result.passphrase_accepted is True  # 音素は一致

    def test_enrollment_with_inconsistent_passphrases_raises_error(
        self, phoneme_auth: PassphraseAuth, _marker: None
    ) -> None:
        """異なるパスフレーズで登録すると PassphraseEnrollmentError。"""
        enroller = phoneme_auth.create_enroller()
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")
        enroller.add_sample(FIXTURES_DIR / "speaker_a_phrase_b.mp3")

        with pytest.raises(PassphraseEnrollmentError):
            enroller.enroll()
