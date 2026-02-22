"""パスフレーズ方式の話者認証。"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from voice_auth_engine.audio_preprocessor import AudioInput, load_audio
from voice_auth_engine.audio_validator import validate_audio
from voice_auth_engine.embedding_extractor import Embedding, extract_embedding
from voice_auth_engine.math import cosine_similarity
from voice_auth_engine.passphrase_validator import validate_passphrase
from voice_auth_engine.speech_detector import detect_speech, extract_speech
from voice_auth_engine.speech_recognizer import transcribe


class PassphraseAuthError(Exception):
    """PassphraseAuth の基底例外。"""


class VerificationResult(NamedTuple):
    """照合結果。"""

    accepted: bool  # 受理/拒否
    score: float  # コサイン類似度 [-1.0, 1.0]


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
    ) -> None:
        """初期化。

        Args:
            threshold: 照合の閾値（コサイン類似度）。
            min_speech_seconds: 最小発話時間（秒）。
            min_unique_phonemes: 音素多様性チェックの最小ユニーク音素数。
                None の場合は音素チェックを無効化。
        """
        self._threshold = threshold
        self._min_speech_seconds = min_speech_seconds
        self._min_unique_phonemes = min_unique_phonemes

    def create_enroller(self) -> PassphraseAuthEnroller:
        """登録用 Enroller を生成する。"""
        return PassphraseAuthEnroller(self)

    def create_verifier(self, embedding: Embedding) -> PassphraseAuthVerifier:
        """照合用 Verifier を生成する。

        Args:
            embedding: 登録済み埋め込みベクトル。
        """
        return PassphraseAuthVerifier(self, embedding)

    def extract_passphrase_embedding(self, audio: AudioInput) -> Embedding:
        """音声入力から検証済み埋め込みベクトルを抽出する。

        音声読み込み → VAD → 発話時間チェック → 音素検証 → 埋め込み抽出。

        Args:
            audio: 音声入力（bytes / str / Path）。

        Returns:
            検証済みの埋め込みベクトル。

        Raises:
            EmptyAudioError: 音声区間が検出されなかった場合。
            InsufficientDurationError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
        """
        audio_data = load_audio(audio)
        segments = detect_speech(audio_data)
        speech = extract_speech(segments)
        validate_audio(speech, min_seconds=self._min_speech_seconds)
        if self._min_unique_phonemes is not None:
            result = transcribe(speech)
            validate_passphrase(result.text, min_unique_phonemes=self._min_unique_phonemes)
        return extract_embedding(speech)


class PassphraseAuthEnroller:
    """声紋登録 (Enroller)。

    音声サンプルを蓄積し、平均埋め込みベクトルを算出する。
    ``PassphraseAuth.create_enroller()`` から生成する。
    """

    def __init__(self, auth: PassphraseAuth) -> None:
        self._auth = auth
        self._embeddings: list[Embedding] = []

    def add_sample(self, audio: AudioInput) -> None:
        """音声サンプルを蓄積する。

        Args:
            audio: 登録用音声データ。

        Raises:
            EmptyAudioError: 音声区間が検出されなかった場合。
            InsufficientDurationError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
        """
        embedding = self._auth.extract_passphrase_embedding(audio)
        self._embeddings.append(embedding)

    def enroll(self) -> Embedding:
        """蓄積サンプルの平均埋め込みベクトルを返す。

        Raises:
            ValueError: サンプルが蓄積されていない場合。
        """
        if not self._embeddings:
            raise ValueError("サンプルが蓄積されていません")
        mean_values = np.mean([e.values for e in self._embeddings], axis=0)
        return Embedding(values=mean_values)

    @property
    def sample_count(self) -> int:
        """蓄積済みサンプル数。"""
        return len(self._embeddings)


class PassphraseAuthVerifier:
    """声紋照合 (Verifier)。

    登録済み埋め込みベクトルに対して音声を照合する。
    ``PassphraseAuth.create_verifier(embedding)`` から生成する。
    """

    def __init__(self, auth: PassphraseAuth, embedding: Embedding) -> None:
        self._auth = auth
        self._embedding = embedding

    def verify(self, audio: AudioInput) -> VerificationResult:
        """音声を登録済み声紋と照合する。

        Args:
            audio: 照合用音声データ。

        Raises:
            EmptyAudioError: 音声区間が検出されなかった場合。
            InsufficientDurationError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
        """
        test_embedding = self._auth.extract_passphrase_embedding(audio)
        score = cosine_similarity(self._embedding.values, test_embedding.values)
        return VerificationResult(accepted=score >= self._auth._threshold, score=score)
