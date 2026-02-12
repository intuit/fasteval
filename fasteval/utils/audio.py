"""Audio utilities for multi-modal evaluation.

This module provides utilities for loading and processing audio files
for speech-to-text and audio understanding evaluation.

Requires: pip install fasteval[audio]
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from fasteval.models.multimodal import AudioInput

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

try:
    from pydub import AudioSegment

    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    AudioSegment = None

try:
    import jiwer

    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    jiwer = None  # type: ignore

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def _check_audio_deps() -> None:
    """Check if audio dependencies are available."""
    if not SOUNDFILE_AVAILABLE and not PYDUB_AVAILABLE:
        raise ImportError(
            "Audio features require the 'audio' extra. "
            "Install with: pip install fasteval[audio]"
        )


def _check_jiwer_deps() -> None:
    """Check if jiwer is available for WER/CER calculation."""
    if not JIWER_AVAILABLE:
        raise ImportError(
            "WER/CER calculation requires jiwer. "
            "Install with: pip install fasteval[audio]"
        )


def load_audio_as_base64(
    source: Union[str, Path, bytes],
    format: str = "auto",
) -> str:
    """
    Load an audio file and return base64-encoded string.

    Args:
        source: Audio source (file path, URL, or raw bytes)
        format: Audio format hint (auto, mp3, wav, etc.)

    Returns:
        Base64-encoded audio string with data URI prefix

    Example:
        b64 = load_audio_as_base64("recording.mp3")
    """
    # Handle bytes directly
    if isinstance(source, bytes):
        encoded = base64.b64encode(source).decode("utf-8")
        mime_type = _guess_audio_mime_type(format)
        return f"data:{mime_type};base64,{encoded}"

    source_str = str(source)

    # Already base64 encoded
    if source_str.startswith("data:"):
        return source_str

    # URL - fetch and encode
    if source_str.startswith(("http://", "https://")):
        return _load_audio_from_url(source_str)

    # File path
    return _load_audio_from_file(source_str)


def _load_audio_from_file(filepath: str) -> str:
    """Load audio from file and return base64."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    with open(path, "rb") as f:
        data = f.read()

    mime_type = _guess_audio_mime_type_from_path(path)
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _load_audio_from_url(url: str) -> str:
    """Load audio from URL and return base64."""
    if not HTTPX_AVAILABLE:
        raise ImportError(
            "URL audio loading requires httpx. " "Install with: pip install httpx"
        )

    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "audio/mpeg")
    if ";" in content_type:
        content_type = content_type.split(";")[0].strip()

    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def _guess_audio_mime_type(format: str) -> str:
    """Guess MIME type from format string."""
    format_lower = format.lower()
    mime_map = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "m4a": "audio/mp4",
        "ogg": "audio/ogg",
        "flac": "audio/flac",
        "webm": "audio/webm",
        "aac": "audio/aac",
        "auto": "audio/mpeg",
    }
    return mime_map.get(format_lower, "audio/mpeg")


def _guess_audio_mime_type_from_path(path: Path) -> str:
    """Guess MIME type from file path."""
    suffix = path.suffix.lower().lstrip(".")
    return _guess_audio_mime_type(suffix)


def get_audio_duration(
    source: Union[str, Path, bytes],
) -> float:
    """
    Get audio duration in seconds.

    Args:
        source: Audio source (file path or bytes)

    Returns:
        Duration in seconds

    Example:
        duration = get_audio_duration("recording.mp3")
    """
    _check_audio_deps()

    if PYDUB_AVAILABLE:
        return _get_duration_pydub(source)
    elif SOUNDFILE_AVAILABLE:
        return _get_duration_soundfile(source)
    else:
        raise ImportError("No audio library available")


def _get_duration_pydub(source: Union[str, Path, bytes]) -> float:
    """Get duration using pydub."""
    import io

    if isinstance(source, bytes):
        audio = AudioSegment.from_file(io.BytesIO(source))
    else:
        audio = AudioSegment.from_file(str(source))

    return audio.duration_seconds


def _get_duration_soundfile(source: Union[str, Path, bytes]) -> float:
    """Get duration using soundfile."""
    import io

    if isinstance(source, bytes):
        data, samplerate = sf.read(io.BytesIO(source))
    else:
        data, samplerate = sf.read(str(source))

    return len(data) / samplerate


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis.

    WER is the standard metric for automatic speech recognition (ASR).
    Lower is better: 0.0 = perfect match, 1.0 = completely different.

    Args:
        reference: Ground truth transcript
        hypothesis: ASR output transcript
        normalize: If True, normalize text (lowercase, remove punctuation)

    Returns:
        WER as a float between 0.0 and 1.0+
        (Can exceed 1.0 if hypothesis is much longer)

    Example:
        wer = calculate_wer(
            reference="The quick brown fox",
            hypothesis="The quik brown fox"
        )
        # wer = 0.25 (1 error in 4 words)
    """
    _check_jiwer_deps()

    if normalize:
        transforms = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
            ]
        )
        return jiwer.wer(
            reference,
            hypothesis,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )

    return jiwer.wer(reference, hypothesis)


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> float:
    """
    Calculate Character Error Rate (CER) between reference and hypothesis.

    CER is useful for character-level ASR evaluation, especially for
    languages without word boundaries.

    Args:
        reference: Ground truth transcript
        hypothesis: ASR output transcript
        normalize: If True, normalize text

    Returns:
        CER as a float between 0.0 and 1.0+

    Example:
        cer = calculate_cer(
            reference="hello world",
            hypothesis="helo world"
        )
        # cer ≈ 0.09 (1 character error in 11 characters)
    """
    _check_jiwer_deps()

    if normalize:
        transforms = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]
        )
        return jiwer.cer(
            reference,
            hypothesis,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )

    return jiwer.cer(reference, hypothesis)


def calculate_mer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
) -> float:
    """
    Calculate Match Error Rate (MER).

    MER is a variant of WER that accounts for the total number of
    words in both reference and hypothesis.

    Args:
        reference: Ground truth transcript
        hypothesis: ASR output transcript
        normalize: If True, normalize text

    Returns:
        MER as a float between 0.0 and 1.0
    """
    _check_jiwer_deps()

    if normalize:
        transforms = jiwer.Compose(
            [
                jiwer.ToLowerCase(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.RemovePunctuation(),
                jiwer.Strip(),
            ]
        )
        return jiwer.mer(
            reference,
            hypothesis,
            reference_transform=transforms,
            hypothesis_transform=transforms,
        )

    return jiwer.mer(reference, hypothesis)


def normalize_audio_input(
    audio: Union[str, Path, bytes, "AudioInput", None],
) -> Optional[str]:
    """
    Normalize any audio input to a base64 data URI.

    Args:
        audio: Audio in any supported format

    Returns:
        Base64 data URI string, or None if input is None
    """
    if audio is None:
        return None

    from fasteval.models.multimodal import AudioInput

    if isinstance(audio, AudioInput):
        return load_audio_as_base64(audio.source, audio.format)

    return load_audio_as_base64(audio)


def convert_wer_to_accuracy(wer: float) -> float:
    """
    Convert WER to accuracy score (0-1 where 1 is perfect).

    Args:
        wer: Word Error Rate (can be > 1.0)

    Returns:
        Accuracy score between 0.0 and 1.0

    Example:
        accuracy = convert_wer_to_accuracy(0.25)  # Returns 0.75
        accuracy = convert_wer_to_accuracy(1.5)   # Returns 0.0
    """
    return max(0.0, 1.0 - wer)


def convert_cer_to_accuracy(cer: float) -> float:
    """
    Convert CER to accuracy score (0-1 where 1 is perfect).

    Args:
        cer: Character Error Rate (can be > 1.0)

    Returns:
        Accuracy score between 0.0 and 1.0
    """
    return max(0.0, 1.0 - cer)
