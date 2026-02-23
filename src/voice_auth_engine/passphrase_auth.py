"""パスフレーズ方式の話者認証。"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from voice_auth_engine.audio_preprocessor import AudioInput, load_audio
from voice_auth_engine.audio_validator import validate_audio
from voice_auth_engine.embedding_extractor import Embedding, extract_embedding
from voice_auth_engine.math import cosine_similarity, normalized_edit_distance
from voice_auth_engine.passphrase_validator import (
    validate_passphrase,
    validate_phoneme_consistency,
)
from voice_auth_engine.phoneme_extractor import Phoneme, extract_phonemes
from voice_auth_engine.speech_detector import detect_speech, extract_speech
from voice_auth_engine.speech_recognizer import transcribe


class PassphraseAuthError(Exception):
    """PassphraseAuth の基底例外。"""


class PassphraseExtractionResult(NamedTuple):
    """パスフレーズ抽出結果。"""

    embedding: Embedding
    phoneme: Phoneme


class EnrollmentResult(NamedTuple):
    """登録結果。"""

    embedding: Embedding
    phoneme: Phoneme  # 基準音素（メドイドで決定）


class VerificationResult(NamedTuple):
    """照合結果。"""

    accepted: bool  # 受理/拒否
    score: float  # コサイン類似度 [-1.0, 1.0]
    phoneme_score: float | None = None  # 正規化編集距離 [0.0, 1.0]
    passphrase_accepted: bool | None = None  # 音素照合の受理/拒否


class PassphraseAuth:
    """パスフレーズ方式の話者認証。

    Enroller と Verifier を生成し、
    音声読み込み → VAD → 発話時間チェック → 音素検証 → 埋め込み抽出の
    共通パイプラインを提供する。

    使用例::

        auth = PassphraseAuth(threshold=0.5)

        # 登録
        enroller = auth.create_enroller()
        enroller.add_sample(audio_bytes_1)
        enroller.add_sample(audio_bytes_2)
        embedding = enroller.enroll()
        saved = embedding.to_bytes()

        # 認証
        embedding = Embedding.from_bytes(saved)
        verifier = auth.create_verifier(embedding)
        result = verifier.verify(audio_bytes)
        result.accepted  # True/False
        result.score     # 0.85
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

    def create_enroller(self) -> PassphraseAuthEnroller:
        """登録用 Enroller を生成する。"""
        return PassphraseAuthEnroller(self)

    def create_verifier(
        self, embedding: Embedding, phoneme: Phoneme | None = None
    ) -> PassphraseAuthVerifier:
        """照合用 Verifier を生成する。

        Args:
            embedding: 登録済み埋め込みベクトル。
            phoneme: 登録済み基準音素。None の場合は音素照合を無効化。
        """
        return PassphraseAuthVerifier(self, embedding, phoneme)

    def extract_passphrase(self, audio: AudioInput) -> PassphraseExtractionResult:
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
            phoneme = extract_phonemes(result.text)
            if self._min_unique_phonemes is not None:
                validate_passphrase(phoneme, min_unique_phonemes=self._min_unique_phonemes)
        else:
            phoneme = Phoneme(values=[])
        embedding = extract_embedding(speech)
        return PassphraseExtractionResult(embedding=embedding, phoneme=phoneme)

    def extract_passphrase_embedding(self, audio: AudioInput) -> Embedding:
        """音声入力から検証済み埋め込みベクトルを抽出する（後方互換）。

        Args:
            audio: 音声入力（bytes / str / Path）。

        Returns:
            検証済みの埋め込みベクトル。

        Raises:
            EmptyAudioError: 音声区間が検出されなかった場合。
            InsufficientDurationError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
        """
        return self.extract_passphrase(audio).embedding


class PassphraseAuthEnroller:
    """声紋登録 (Enroller)。

    音声サンプルを蓄積し、平均埋め込みベクトルと基準音素列を算出する。
    ``PassphraseAuth.create_enroller()`` から生成する。
    """

    def __init__(self, auth: PassphraseAuth) -> None:
        self._auth = auth
        self._embeddings: list[Embedding] = []
        self._phoneme_samples: list[Phoneme] = []

    def add_sample(self, audio: AudioInput) -> None:
        """音声サンプルを蓄積する。

        Args:
            audio: 登録用音声データ。

        Raises:
            EmptyAudioError: 音声区間が検出されなかった場合。
            InsufficientDurationError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
        """
        result = self._auth.extract_passphrase(audio)
        self._embeddings.append(result.embedding)
        self._phoneme_samples.append(result.phoneme)

    def enroll(self) -> EnrollmentResult:
        """蓄積サンプルの平均埋め込みベクトルと基準音素列を返す。

        Raises:
            ValueError: サンプルが蓄積されていない場合。
            PassphraseEnrollmentError: 音素列の整合性チェックに失敗した場合。
        """
        if not self._embeddings:
            raise ValueError("サンプルが蓄積されていません")
        mean_values = np.mean([e.values for e in self._embeddings], axis=0)
        embedding = Embedding(values=mean_values)
        if self._auth._phoneme_threshold is not None:
            validate_phoneme_consistency(
                self._phoneme_samples, threshold=self._auth._phoneme_threshold
            )
            phoneme = Phoneme.select_reference(self._phoneme_samples)
        else:
            phoneme = Phoneme(values=[])
        return EnrollmentResult(embedding=embedding, phoneme=phoneme)

    @property
    def sample_count(self) -> int:
        """蓄積済みサンプル数。"""
        return len(self._embeddings)


class PassphraseAuthVerifier:
    """声紋照合 (Verifier)。

    登録済み埋め込みベクトルに対して音声を照合する。
    ``PassphraseAuth.create_verifier(embedding)`` から生成する。
    """

    def __init__(
        self,
        auth: PassphraseAuth,
        embedding: Embedding,
        phoneme: Phoneme | None = None,
    ) -> None:
        self._auth = auth
        self._embedding = embedding
        self._phoneme = phoneme

    def verify(self, audio: AudioInput) -> VerificationResult:
        """音声を登録済み声紋と照合する。

        Args:
            audio: 照合用音声データ。

        Raises:
            EmptyAudioError: 音声区間が検出されなかった場合。
            InsufficientDurationError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
        """
        result = self._auth.extract_passphrase(audio)
        score = cosine_similarity(self._embedding.values, result.embedding.values)
        speaker_accepted = score >= self._auth.threshold

        if self._phoneme is not None and self._auth._phoneme_threshold is not None:
            phoneme_score = normalized_edit_distance(self._phoneme.values, result.phoneme.values)
            passphrase_accepted = phoneme_score <= self._auth._phoneme_threshold
            accepted = speaker_accepted and passphrase_accepted
            return VerificationResult(
                accepted=accepted,
                score=score,
                phoneme_score=phoneme_score,
                passphrase_accepted=passphrase_accepted,
            )

        return VerificationResult(accepted=speaker_accepted, score=score)
