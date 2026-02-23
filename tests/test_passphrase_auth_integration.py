"""passphrase_auth モジュールの統合テスト（実音声）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from voice_auth_engine.audio_validator import EmptyAudioError, InsufficientDurationError
from voice_auth_engine.passphrase_auth import PassphraseAuth

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
