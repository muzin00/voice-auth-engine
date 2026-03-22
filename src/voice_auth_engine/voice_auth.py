"""話者認証。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

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


@dataclass(frozen=True)
class Passphrase:
    """パスフレーズ抽出結果。"""

    embedding: Embedding
    phoneme: Phoneme
    transcription: str  # 音声認識による書き起こしテキスト
    speech_duration: float  # VAD後の発話区間の長さ（秒）


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
        passphrases = [auth.extract_passphrase(a) for a in audios]
        selected, index = auth.select_passphrase(passphrases)
        saved = selected.embedding.to_bytes()

        # 認証
        passphrase = auth.extract_passphrase(audio_bytes)
        result = auth.verify_passphrase(passphrase, selected)
        result.voiceprint_accepted  # True/False
        result.voiceprint_score     # 0.85
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

    def select_passphrase(self, passphrases: list[Passphrase]) -> tuple[Passphrase, int]:
        """抽出済みパスフレーズから最適な1サンプルを選択する。

        embedding のコサイン距離に基づく medoid 選択で、他サンプルとの距離の
        総和が最小のサンプルを選ぶ。サンプルが1つの場合はそのまま返す。

        Args:
            passphrases: extract_passphrase() で抽出済みのパスフレーズのリスト。

        Returns:
            選択されたパスフレーズとそのインデックス。

        Raises:
            ValueError: passphrases が空の場合。
            PhonemeConsistencyError: 音素列の整合性チェックに失敗した場合。
        """
        if not passphrases:
            raise ValueError("passphrases が空です")
        phoneme_samples = [p.phoneme for p in passphrases]
        if self._phoneme_threshold is not None:
            validate_phoneme_consistency(phoneme_samples, threshold=self._phoneme_threshold)
        if len(passphrases) == 1:
            return passphrases[0], 0
        vectors = [p.embedding.values for p in passphrases]
        distances = cosine_distance_matrix(vectors)
        index = select_medoid(distances)
        return passphrases[index], index

    def verify_passphrase(
        self,
        passphrase: Passphrase,
        enrolled: Passphrase,
    ) -> VerificationResult:
        """抽出済みパスフレーズを登録済み声紋と照合する。

        Args:
            passphrase: extract_passphrase() で抽出済みのパスフレーズ。
            enrolled: select_passphrase() で選択された登録済みパスフレーズ。
        """
        score = cosine_similarity(enrolled.embedding.values, passphrase.embedding.values)
        speaker_accepted = score >= self._threshold

        if self._phoneme_threshold is not None:
            passphrase_score = normalized_edit_distance(
                enrolled.phoneme.values, passphrase.phoneme.values
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

    def extract_passphrase(self, audio: AudioInput) -> Passphrase:
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
            transcription = result.text
            phoneme = extract_phonemes(result.text)
            if self._min_unique_phonemes is not None:
                validate_passphrase(phoneme, min_unique_phonemes=self._min_unique_phonemes)
        else:
            transcription = ""
            phoneme = Phoneme(values=[])
        embedding = extract_embedding(speech)
        return Passphrase(
            embedding=embedding,
            phoneme=phoneme,
            transcription=transcription,
            speech_duration=speech.duration,
        )
