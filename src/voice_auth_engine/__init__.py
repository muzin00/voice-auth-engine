from voice_auth_engine.audio_preprocessor import (
    AudioData,
    AudioDecodeError,
    AudioInput,
    AudioPreprocessError,
    decode_audio,
    load_audio,
)
from voice_auth_engine.audio_validator import (
    DEFAULT_MIN_SPEECH_SECONDS,
    SUPPORTED_EXTENSIONS,
    AudioValidationError,
    EmptyAudioError,
    InsufficientDurationError,
    UnsupportedExtensionError,
    validate_audio,
    validate_extension,
)
from voice_auth_engine.embedding_extractor import (
    Embedding,
    EmbeddingExtractionError,
    EmbeddingExtractorError,
    EmbeddingModelLoadError,
    extract_embedding,
)
from voice_auth_engine.math import cosine_similarity, normalized_edit_distance
from voice_auth_engine.model_config import ModelConfig
from voice_auth_engine.model_downloader import ModelDownloader, ModelDownloadError
from voice_auth_engine.phoneme_extractor import (
    Phoneme,
    extract_phonemes,
)
from voice_auth_engine.phoneme_validator import (
    EmptyPhonemeError,
    InsufficientPhonemeError,
    PhonemeConsistencyError,
    PhonemeValidationError,
    validate_phoneme,
    validate_phoneme_consistency,
)
from voice_auth_engine.speech_detector import (
    SpeechDetectorError,
    SpeechDetectorModelLoadError,
    SpeechSegment,
    SpeechSegments,
    detect_speech,
    extract_speech,
)
from voice_auth_engine.speech_recognizer import (
    RecognitionError,
    RecognizerModelLoadError,
    SpeechRecognizerError,
    TranscriptionResult,
    transcribe,
)
from voice_auth_engine.voice_auth import (
    ExtractionResult,
    VerificationResult,
    VoiceAuth,
    VoiceAuthError,
    VoiceInput,
)

__all__ = [
    "AudioData",
    "AudioDecodeError",
    "AudioInput",
    "AudioPreprocessError",
    "AudioValidationError",
    "DEFAULT_MIN_SPEECH_SECONDS",
    "Embedding",
    "EmbeddingExtractionError",
    "EmbeddingExtractorError",
    "EmbeddingModelLoadError",
    "EmptyAudioError",
    "EmptyPhonemeError",
    "ModelDownloadError",
    "ModelDownloader",
    "InsufficientDurationError",
    "InsufficientPhonemeError",
    "VoiceAuth",
    "VoiceAuthError",
    "PhonemeConsistencyError",
    "ExtractionResult",
    "VoiceInput",
    "VerificationResult",
    "Phoneme",
    "PhonemeValidationError",
    "RecognitionError",
    "RecognizerModelLoadError",
    "SpeechDetectorError",
    "SpeechDetectorModelLoadError",
    "SpeechRecognizerError",
    "SpeechSegment",
    "SpeechSegments",
    "SUPPORTED_EXTENSIONS",
    "TranscriptionResult",
    "UnsupportedExtensionError",
    "ModelConfig",
    "extract_phonemes",
    "cosine_similarity",
    "normalized_edit_distance",
    "detect_speech",
    "extract_embedding",
    "extract_speech",
    "load_audio",
    "decode_audio",
    "transcribe",
    "validate_audio",
    "validate_extension",
    "validate_phoneme",
    "validate_phoneme_consistency",
]
