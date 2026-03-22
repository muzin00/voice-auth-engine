"""話者認証。"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

from voice_auth_engine.audio_preprocessor import AudioInput, load_audio
from voice_auth_engine.audio_validator import validate_audio
from voice_auth_engine.embedding_extractor import Embedding, extract_embedding
from voice_auth_engine.math import (
    cosine_distance_matrix,
    cosine_similarity,
    normalized_edit_distance,
    select_medoid,
)
from voice_auth_engine.passphrase_validator import (
    validate_passphrase,
    validate_phoneme_consistency,
)
from voice_auth_engine.phoneme_extractor import Phoneme, extract_phonemes
from voice_auth_engine.speech_detector import detect_speech, extract_speech
from voice_auth_engine.speech_recognizer import transcribe


class VoiceAuthError(Exception):
    """VoiceAuth の基底例外。"""


class VoiceInput(TypedDict):
    """select_best_voice / verify の入力型。"""

    embedding_values: list[float]
    phoneme_values: list[str]


class ExtractionResult(TypedDict):
    """extract の戻り値型。"""

    embedding_values: list[float]
    phoneme_values: list[str]
    transcription: str
    speech_duration: float


class VerificationResult(NamedTuple):
    """照合結果。"""

    voiceprint_accepted: bool  # 受理/拒否
    voiceprint_score: float  # コサイン類似度 [-1.0, 1.0]
    passphrase_accepted: bool | None = None  # 音素照合の受理/拒否
    passphrase_score: float | None = None  # 正規化編集距離 [0.0, 1.0]


class VoiceAuth:
    """話者認証。

    音声読み込み → VAD → 発話時間チェック → 音素検証 → 埋め込み抽出の
    パイプラインと、選択・照合メソッドを提供する。

    使用例::

        auth = VoiceAuth(threshold=0.5)

        # 登録
        results = [auth.extract_audio(a) for a in audios]
        voices = [{"embedding_values": r["embedding_values"],
                    "phoneme_values": r["phoneme_values"]} for r in results]
        selected, index = auth.select_best_voice(voices)

        # 認証
        result = auth.extract_audio(audio_bytes)
        voice = {"embedding_values": result["embedding_values"],
                 "phoneme_values": result["phoneme_values"]}
        verification = auth.verify_voice(voice, selected)
        verification.voiceprint_accepted  # True/False
        verification.voiceprint_score     # 0.85
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_seconds: float = 3.0,
        min_unique_phonemes: int | None = 5,
        phoneme_threshold: float | None = None,
    ) -> None:
        """初期化。

        Args:
            threshold: 照合の閾値（コサイン類似度）。
            min_speech_seconds: 最小発話時間（秒）。
            min_unique_phonemes: 音素多様性チェックの最小ユニーク音素数。
                None の場合は音素チェックを無効化。
            phoneme_threshold: 登録時の音素列整合性チェック閾値（正規化編集距離）。
                None の場合は音素整合性チェックを無効化。
        """
        self._threshold = threshold
        self._min_speech_seconds = min_speech_seconds
        self._min_unique_phonemes = min_unique_phonemes
        self._phoneme_threshold = phoneme_threshold

    @property
    def threshold(self) -> float:
        """照合の閾値（コサイン類似度）。"""
        return self._threshold

    def select_best_voice(self, voices: list[VoiceInput]) -> tuple[VoiceInput, int]:
        """入力音声から最適な1サンプルを選択する。

        embedding のコサイン距離に基づく medoid 選択で、他サンプルとの距離の
        総和が最小のサンプルを選ぶ。サンプルが1つの場合はそのまま返す。

        Args:
            voices: 音声入力のリスト。

        Returns:
            選択された音声入力とそのインデックス。

        Raises:
            ValueError: voices が空の場合。
            PhonemeConsistencyError: 音素列の整合性チェックに失敗した場合。
        """
        if not voices:
            raise ValueError("voices が空です")
        if self._phoneme_threshold is not None:
            phoneme_samples = [Phoneme(values=v["phoneme_values"]) for v in voices]
            validate_phoneme_consistency(phoneme_samples, threshold=self._phoneme_threshold)
        if len(voices) == 1:
            return voices[0], 0
        vectors = [Embedding.from_floats(v["embedding_values"]).values for v in voices]
        distances = cosine_distance_matrix(vectors)
        index = select_medoid(distances)
        return voices[index], index

    def verify_voice(
        self,
        target: VoiceInput,
        reference: VoiceInput,
    ) -> VerificationResult:
        """音声入力を登録済み声紋と照合する。

        Args:
            target: 認証対象の音声入力。
            reference: 登録済みの音声入力。
        """
        target_embedding = Embedding.from_floats(target["embedding_values"])
        reference_embedding = Embedding.from_floats(reference["embedding_values"])
        score = cosine_similarity(reference_embedding.values, target_embedding.values)
        speaker_accepted = score >= self._threshold

        if self._phoneme_threshold is not None:
            passphrase_score = normalized_edit_distance(
                reference["phoneme_values"], target["phoneme_values"]
            )
            passphrase_accepted = passphrase_score <= self._phoneme_threshold
            voiceprint_accepted = speaker_accepted and passphrase_accepted
            return VerificationResult(
                voiceprint_accepted=voiceprint_accepted,
                voiceprint_score=score,
                passphrase_score=passphrase_score,
                passphrase_accepted=passphrase_accepted,
            )

        return VerificationResult(voiceprint_accepted=speaker_accepted, voiceprint_score=score)

    def extract_audio(self, audio: AudioInput) -> ExtractionResult:
        """音声入力から検証済み埋め込みベクトルと音素列を抽出する。

        音声読み込み → VAD → 発話時間チェック → 音素検証 → 埋め込み抽出。

        Args:
            audio: 音声入力（bytes / str / Path）。

        Returns:
            埋め込みベクトルと音素列。

        Raises:
            EmptyAudioError: 音声区間が検出されなかった場合。
            InsufficientDurationError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
        """
        audio_data = load_audio(audio)
        segments = detect_speech(audio_data)
        speech = extract_speech(segments)
        validate_audio(speech, min_seconds=self._min_speech_seconds)
        if self._min_unique_phonemes is not None or self._phoneme_threshold is not None:
            result = transcribe(speech)
            transcription_text = result.text
            phoneme = extract_phonemes(result.text)
            if self._min_unique_phonemes is not None:
                validate_passphrase(phoneme, min_unique_phonemes=self._min_unique_phonemes)
        else:
            transcription_text = ""
            phoneme = Phoneme(values=[])
        embedding = extract_embedding(speech)
        return ExtractionResult(
            embedding_values=list(embedding.values),
            phoneme_values=phoneme.values,
            transcription=transcription_text,
            speech_duration=speech.duration,
        )
