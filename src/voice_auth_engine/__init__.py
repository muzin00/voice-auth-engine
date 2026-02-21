from voice_auth_engine.audio_preprocessor import (
    AudioData,
    AudioDecodeError,
    AudioPreprocessError,
    UnsupportedFormatError,
    load_audio,
)
from voice_auth_engine.embedding_extractor import (
    Embedding,
    EmbeddingExtractionError,
    EmbeddingExtractorError,
    EmbeddingModelLoadError,
    extract_embedding,
)
from voice_auth_engine.passphrase_validator import (
    EmptyPassphraseError,
    InsufficientPhonemeError,
    PassphraseInfo,
    PassphraseValidationError,
    analyze_passphrase,
    validate_passphrase,
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

__all__ = [
    "AudioData",
    "AudioDecodeError",
    "AudioPreprocessError",
    "Embedding",
    "EmbeddingExtractionError",
    "EmbeddingExtractorError",
    "EmbeddingModelLoadError",
    "EmptyPassphraseError",
    "InsufficientPhonemeError",
    "PassphraseInfo",
    "PassphraseValidationError",
    "RecognitionError",
    "RecognizerModelLoadError",
    "SpeechDetectorError",
    "SpeechDetectorModelLoadError",
    "SpeechRecognizerError",
    "SpeechSegment",
    "SpeechSegments",
    "TranscriptionResult",
    "UnsupportedFormatError",
    "analyze_passphrase",
    "detect_speech",
    "extract_embedding",
    "extract_speech",
    "load_audio",
    "transcribe",
    "validate_passphrase",
]
