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
from voice_auth_engine.speech_recognizer import (
    RecognitionError,
    RecognizerModelLoadError,
    SpeechRecognizerError,
    TranscriptionResult,
    transcribe,
)
from voice_auth_engine.vad import (
    SpeechSegment,
    SpeechSegments,
    VadError,
    VadModelLoadError,
    detect_speech,
    extract_speech,
)

__all__ = [
    "AudioData",
    "AudioDecodeError",
    "AudioPreprocessError",
    "Embedding",
    "EmbeddingExtractionError",
    "EmbeddingExtractorError",
    "EmbeddingModelLoadError",
    "RecognitionError",
    "RecognizerModelLoadError",
    "SpeechRecognizerError",
    "SpeechSegment",
    "SpeechSegments",
    "TranscriptionResult",
    "UnsupportedFormatError",
    "VadError",
    "VadModelLoadError",
    "detect_speech",
    "extract_embedding",
    "extract_speech",
    "load_audio",
    "transcribe",
]
