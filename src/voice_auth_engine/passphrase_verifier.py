"""パスフレーズ方式による話者認証。"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from voice_auth_engine.audio_preprocessor import AudioData
from voice_auth_engine.embedding_extractor import Embedding, extract_embedding
from voice_auth_engine.math import cosine_similarity
from voice_auth_engine.passphrase_validator import validate_passphrase
from voice_auth_engine.speech_recognizer import transcribe


class PassphraseVerifierError(Exception):
    """パスフレーズ認証の基底例外。"""


class InsufficientSpeechError(PassphraseVerifierError):
    """発話時間が不足。"""


class VerificationResult(NamedTuple):
    """照合結果。"""

    accepted: bool  # 受理/拒否
    score: float  # コサイン類似度 [-1.0, 1.0]


def check_speech_duration(
    audio: AudioData,
    *,
    min_seconds: float = 3.0,
) -> None:
    """音声の発話時間が十分か検証する。

    Args:
        audio: 検証対象の音声データ（VAD 済みを想定）。
        min_seconds: 必要な最小発話時間（秒）。

    Raises:
        InsufficientSpeechError: 発話時間が min_seconds 未満の場合。
    """
    duration = len(audio.samples) / audio.sample_rate
    if duration < min_seconds:
        raise InsufficientSpeechError(
            f"発話時間が不足しています: {duration:.1f}秒 < {min_seconds:.1f}秒"
        )


class PassphraseEnroller:
    """パスフレーズ方式の声紋登録エンジン。

    音声サンプルを蓄積し、登録確定時に PassphraseVerifier を返す。
    呼び出し側で VAD（detect_speech → extract_speech）済みの
    AudioData を渡すことを想定。

    使用例::

        enroller = PassphraseEnroller()
        for audio in audio_list:
            enroller.add_sample(audio)
        verifier = enroller.enroll()
        result = verifier.verify(test_audio)
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_seconds: float = 3.0,
        min_unique_phonemes: int | None = 5,
        embedding_model_path: str | Path | None = None,
        asr_model_dir: str | Path | None = None,
    ) -> None:
        """初期化。

        Args:
            threshold: 照合の閾値（コサイン類似度）。
                生成される PassphraseVerifier に引き継がれる。
            min_speech_seconds: 最小発話時間（秒）。
            min_unique_phonemes: 音素多様性チェックの最小ユニーク音素数。
                None の場合は音素チェックを無効化。
            embedding_model_path: CAM++ モデルファイルのパス。
            asr_model_dir: SenseVoice モデルディレクトリのパス。
        """
        self._threshold = threshold
        self._min_speech_seconds = min_speech_seconds
        self._min_unique_phonemes = min_unique_phonemes
        self._embedding_model_path = embedding_model_path
        self._asr_model_dir = asr_model_dir
        self._embeddings: list[Embedding] = []

    def add_sample(self, audio: AudioData) -> None:
        """音声サンプルを蓄積する。

        Args:
            audio: 登録用音声データ（VAD 済み）。

        Raises:
            InsufficientSpeechError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
            EmbeddingExtractionError: 埋め込み抽出に失敗した場合。
        """
        check_speech_duration(audio, min_seconds=self._min_speech_seconds)
        if self._min_unique_phonemes is not None:
            result = transcribe(audio, model_dir=self._asr_model_dir)
            validate_passphrase(result.text, min_unique_phonemes=self._min_unique_phonemes)
        embedding = extract_embedding(audio, model_path=self._embedding_model_path)
        self._embeddings.append(embedding)

    def enroll(self) -> PassphraseVerifier:
        """蓄積されたサンプルから声紋を登録し、照合エンジンを返す。

        Returns:
            PassphraseVerifier: 登録済み声紋を持つパスフレーズ照合エンジン。

        Raises:
            ValueError: サンプルが1つも蓄積されていない場合。
        """
        if not self._embeddings:
            raise ValueError("サンプルが蓄積されていません")
        mean_values = np.mean([e.values for e in self._embeddings], axis=0)
        mean_embedding = Embedding(values=mean_values)
        return PassphraseVerifier(
            mean_embedding,
            threshold=self._threshold,
            min_speech_seconds=self._min_speech_seconds,
            min_unique_phonemes=self._min_unique_phonemes,
            embedding_model_path=self._embedding_model_path,
            asr_model_dir=self._asr_model_dir,
        )

    @property
    def sample_count(self) -> int:
        """蓄積済みサンプル数を返す。"""
        return len(self._embeddings)


class PassphraseVerifier:
    """パスフレーズ方式の話者照合エンジン。

    PassphraseEnroller.enroll() から生成される。
    登録済みの声紋埋め込みベクトルを保持し、パスフレーズの音素多様性検証と
    声紋照合を行う。
    """

    def __init__(
        self,
        embedding: Embedding,
        *,
        threshold: float = 0.5,
        min_speech_seconds: float = 3.0,
        min_unique_phonemes: int | None = 5,
        embedding_model_path: str | Path | None = None,
        asr_model_dir: str | Path | None = None,
    ) -> None:
        """初期化。

        Args:
            embedding: 登録済みの平均埋め込みベクトル。
            threshold: 照合の閾値（コサイン類似度）。
            min_speech_seconds: 最小発話時間（秒）。
            min_unique_phonemes: 音素多様性チェックの最小ユニーク音素数。
                None の場合は音素チェックを無効化。
            embedding_model_path: CAM++ モデルファイルのパス。
            asr_model_dir: SenseVoice モデルディレクトリのパス。
        """
        self._embedding = embedding
        self._threshold = threshold
        self._min_speech_seconds = min_speech_seconds
        self._min_unique_phonemes = min_unique_phonemes
        self._embedding_model_path = embedding_model_path
        self._asr_model_dir = asr_model_dir

    def verify(self, audio: AudioData) -> VerificationResult:
        """登録済み声紋とパスフレーズ方式で照合する。

        Args:
            audio: 照合用音声データ（VAD 済み）。

        Returns:
            VerificationResult: 照合結果（accepted, score）。

        Raises:
            InsufficientSpeechError: 発話時間が不足の場合。
            InsufficientPhonemeError: 音素多様性が不足の場合。
            EmbeddingExtractionError: 埋め込み抽出に失敗した場合。
        """
        check_speech_duration(audio, min_seconds=self._min_speech_seconds)
        if self._min_unique_phonemes is not None:
            result = transcribe(audio, model_dir=self._asr_model_dir)
            validate_passphrase(result.text, min_unique_phonemes=self._min_unique_phonemes)
        embedding = extract_embedding(audio, model_path=self._embedding_model_path)
        score = cosine_similarity(self._embedding.values, embedding.values)
        return VerificationResult(accepted=score >= self._threshold, score=score)

    @property
    def threshold(self) -> float:
        """照合の閾値を返す。"""
        return self._threshold
