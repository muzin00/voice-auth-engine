"""voice_auth モジュールの統合テスト（実音声）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voice_auth_engine.audio_validator import EmptyAudioError, InsufficientDurationError
from voice_auth_engine.embedding_extractor import Embedding
from voice_auth_engine.phoneme_validator import PhonemeConsistencyError
from voice_auth_engine.voice_auth import ExtractionResult, VoiceAuth, VoiceInput

from .audio_factory import generate_silence_samples, make_audio_data

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _to_voice_input(result: ExtractionResult) -> VoiceInput:
    """ExtractionResult から VoiceInput を作成するヘルパー。"""
    return VoiceInput(
        embedding_values=result["embedding_values"],
        phoneme_values=result["phoneme_values"],
    )


@pytest.fixture
def auth() -> VoiceAuth:
    """統合テスト用 VoiceAuth。"""
    return VoiceAuth(threshold=0.8)


class TestVoiceAuthIntegration:
    """実音声を使った話者認証の統合テスト。"""

    def test_enroll_and_verify_same_speaker(self, auth: VoiceAuth) -> None:
        """登録→本人認証で accepted=True。"""
        voices = [_to_voice_input(auth.extract_audio(FIXTURES_DIR / "speaker_a_enroll.mp3"))]
        selected, _ = auth.select_best_voice(voices)

        target = _to_voice_input(auth.extract_audio(FIXTURES_DIR / "speaker_a_verify.mp3"))
        result = auth.verify_voice(target, selected)
        assert result.voiceprint_accepted is True
        assert result.voiceprint_score > auth.threshold

    @pytest.mark.parametrize(
        "other_speaker",
        ["speaker_b_verify.mp3", "speaker_c_verify.mp3", "speaker_d_verify.mp3"],
    )
    def test_reject_different_speaker(self, auth: VoiceAuth, other_speaker: str) -> None:
        """登録→他人認証で accepted=False。"""
        voices = [_to_voice_input(auth.extract_audio(FIXTURES_DIR / "speaker_a_enroll.mp3"))]
        selected, _ = auth.select_best_voice(voices)

        target = _to_voice_input(auth.extract_audio(FIXTURES_DIR / other_speaker))
        result = auth.verify_voice(target, selected)
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
        voices = [_to_voice_input(auth.extract_audio(FIXTURES_DIR / "speaker_a_enroll.mp3"))]
        selected, _ = auth.select_best_voice(voices)

        same_target = _to_voice_input(auth.extract_audio(FIXTURES_DIR / "speaker_a_verify.mp3"))
        diff_target = _to_voice_input(auth.extract_audio(FIXTURES_DIR / other_speaker))
        same_result = auth.verify_voice(same_target, selected)
        diff_result = auth.verify_voice(diff_target, selected)
        assert same_result.voiceprint_score > diff_result.voiceprint_score

    def test_embedding_serialization_roundtrip(self, auth: VoiceAuth) -> None:
        """埋め込みベクトルのシリアライズ→デシリアライズ後も認証可能。"""
        result = auth.extract_audio(FIXTURES_DIR / "speaker_a_enroll.mp3")
        voices = [_to_voice_input(result)]
        selected, _ = auth.select_best_voice(voices)

        values_array = np.array(selected["embedding_values"], dtype=np.float32)
        restored = Embedding.from_bytes(Embedding(values=values_array).to_bytes())
        restored_ref: VoiceInput = {
            "embedding_values": list(restored.values),
            "phoneme_values": selected["phoneme_values"],
        }
        target = _to_voice_input(auth.extract_audio(FIXTURES_DIR / "speaker_a_verify.mp3"))
        result = auth.verify_voice(target, restored_ref)
        assert result.voiceprint_accepted is True

    def test_silence_raises_empty_audio_error(self, auth: VoiceAuth) -> None:
        """無音音声で EmptyAudioError が発生する。"""
        silence = make_audio_data(generate_silence_samples(duration=5.0))
        with pytest.raises(EmptyAudioError):
            auth.extract_audio(silence)

    def test_short_speech_raises_insufficient_duration_error(
        self,
        auth: VoiceAuth,
    ) -> None:
        """発話時間が短い音声で InsufficientDurationError が発生する。"""
        with pytest.raises(InsufficientDurationError):
            auth.extract_audio(FIXTURES_DIR / "digits_clear.mp3")


@pytest.fixture
def phoneme_auth() -> VoiceAuth:
    """音素照合を有効にした統合テスト用 VoiceAuth。"""
    return VoiceAuth(threshold=0.8, phoneme_threshold=0.3)


class TestPhonemeVerificationIntegration:
    """音素ベースの照合の統合テスト。"""

    def test_select_with_phoneme_threshold(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """phoneme_threshold 有効で選択すると phoneme_values を含む VoiceInput が返る。"""
        voices = [
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")),
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")),
        ]
        selected, index = phoneme_auth.select_best_voice(voices)

        assert len(selected["phoneme_values"]) > 0
        assert selected is voices[index]

    def test_select_three_samples(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """3サンプルで medoid が選択される。"""
        voices = [
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")),
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")),
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_3.mp3")),
        ]
        selected, index = phoneme_auth.select_best_voice(voices)

        assert len(selected["phoneme_values"]) > 0
        assert 0 <= index <= 2

    def test_same_speaker_same_phoneme_accepted(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """本人+同一音素で accepted=True, phoneme_accepted=True。"""
        voices = [
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")),
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")),
        ]
        selected, _ = phoneme_auth.select_best_voice(voices)

        target = _to_voice_input(
            phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_verify.mp3")
        )
        result = phoneme_auth.verify_voice(target, selected)

        assert result.voiceprint_accepted is True
        assert result.phoneme_accepted is True
        assert result.phoneme_score is not None
        assert phoneme_auth._phoneme_threshold is not None
        assert result.phoneme_score <= phoneme_auth._phoneme_threshold

    def test_same_speaker_different_phoneme_rejected(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """本人+異なる音素で accepted=False（話者OK, 音素NG）。"""
        voices = [
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")),
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")),
        ]
        selected, _ = phoneme_auth.select_best_voice(voices)

        target = _to_voice_input(
            phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_b.mp3")
        )
        result = phoneme_auth.verify_voice(target, selected)

        assert result.voiceprint_accepted is False
        assert result.voiceprint_score >= phoneme_auth.threshold  # 話者は本人
        assert result.phoneme_accepted is False  # 音素が不一致

    def test_different_speaker_same_phoneme_rejected(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """他人+同一音素で accepted=False（話者NG, 音素OK）。"""
        voices = [
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")),
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_2.mp3")),
        ]
        selected, _ = phoneme_auth.select_best_voice(voices)

        target = _to_voice_input(
            phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_b_phrase_a.mp3")
        )
        result = phoneme_auth.verify_voice(target, selected)

        assert result.voiceprint_accepted is False
        assert result.voiceprint_score < phoneme_auth.threshold  # 話者が異なる
        assert result.phoneme_accepted is True  # 音素は一致

    def test_inconsistent_phonemes_raises_error(
        self,
        phoneme_auth: VoiceAuth,
    ) -> None:
        """異なる音素で登録すると PhonemeConsistencyError。"""
        voices = [
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_a_1.mp3")),
            _to_voice_input(phoneme_auth.extract_audio(FIXTURES_DIR / "speaker_a_phrase_b.mp3")),
        ]
        with pytest.raises(PhonemeConsistencyError):
            phoneme_auth.select_best_voice(voices)
