"""Multi-modal data models for fasteval.

Supports evaluation of vision-language models, audio/speech systems,
and image generation models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ImageInput(BaseModel):
    """
    Represents an image for evaluation.

    Supports multiple input formats:
    - File path (str or Path)
    - URL (http:// or https://)
    - Base64-encoded data
    - Raw bytes

    Example:
        # From file path
        image = ImageInput(source="path/to/image.png")

        # From URL
        image = ImageInput(source="https://example.com/image.jpg")

        # From base64
        image = ImageInput(source="data:image/png;base64,iVBOR...", format="base64")

        # From bytes
        image = ImageInput(source=raw_bytes, format="bytes")
    """

    source: Union[str, Path, bytes]
    format: str = "auto"  # auto, png, jpg, jpeg, webp, gif, base64, url, bytes
    width: Optional[int] = None
    height: Optional[int] = None
    alt_text: Optional[str] = None  # Description for accessibility/context
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("source", mode="before")
    @classmethod
    def convert_path(cls, v: Any) -> Union[str, bytes]:
        if isinstance(v, Path):
            return str(v)
        return v

    def is_url(self) -> bool:
        """Check if source is a URL."""
        if isinstance(self.source, str):
            return self.source.startswith(("http://", "https://"))
        return False

    def is_base64(self) -> bool:
        """Check if source is base64 encoded."""
        if isinstance(self.source, str):
            return self.source.startswith("data:") or self.format == "base64"
        return False

    def is_file(self) -> bool:
        """Check if source is a file path."""
        if isinstance(self.source, str) and not self.is_url() and not self.is_base64():
            return Path(self.source).exists()
        return False


class AudioInput(BaseModel):
    """
    Represents audio for evaluation.

    Supports multiple input formats:
    - File path (str or Path)
    - URL (http:// or https://)
    - Raw bytes

    Example:
        # From file path
        audio = AudioInput(source="recording.mp3")

        # From URL
        audio = AudioInput(source="https://example.com/audio.wav")

        # With duration info
        audio = AudioInput(source="speech.wav", duration_seconds=120.5)
    """

    source: Union[str, Path, bytes]
    format: str = "auto"  # auto, mp3, wav, m4a, ogg, flac, bytes
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    transcript: Optional[str] = None  # Ground truth transcript if available
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("source", mode="before")
    @classmethod
    def convert_path(cls, v: Any) -> Union[str, bytes]:
        if isinstance(v, Path):
            return str(v)
        return v

    def is_url(self) -> bool:
        """Check if source is a URL."""
        if isinstance(self.source, str):
            return self.source.startswith(("http://", "https://"))
        return False

    def is_file(self) -> bool:
        """Check if source is a file path."""
        if isinstance(self.source, str) and not self.is_url():
            return Path(self.source).exists()
        return False


class MultiModalContext(BaseModel):
    """
    Context containing multiple modalities for RAG and retrieval evaluation.

    Represents retrieved or provided context that may include text,
    images, and audio - useful for multi-modal RAG evaluation.

    Example:
        context = MultiModalContext(
            text="The chart shows quarterly revenue growth.",
            images=[ImageInput(source="chart.png")],
        )
    """

    text: Optional[str] = None
    images: List[ImageInput] = Field(default_factory=list)
    audio: List[AudioInput] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_empty(self) -> bool:
        """Check if context has no content."""
        return not self.text and not self.images and not self.audio

    def has_images(self) -> bool:
        """Check if context contains images."""
        return len(self.images) > 0

    def has_audio(self) -> bool:
        """Check if context contains audio."""
        return len(self.audio) > 0


class GeneratedImage(BaseModel):
    """
    Represents a generated image for image generation evaluation.

    Used specifically for evaluating image generation models like
    DALL-E, Midjourney, Stable Diffusion.

    Example:
        generated = GeneratedImage(
            image=ImageInput(source="output.png"),
            prompt="A red sports car on a mountain road",
            model="dall-e-3",
        )
    """

    image: ImageInput
    prompt: str
    model: Optional[str] = None
    seed: Optional[int] = None
    generation_params: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
