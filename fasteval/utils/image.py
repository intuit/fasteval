"""Image utilities for multi-modal evaluation.

This module provides utilities for loading, encoding, and processing images
for vision-language model evaluation.

Requires: pip install fasteval[vision]
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from fasteval.models.multimodal import ImageInput

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from PIL import Image

    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    Image = None  # type: ignore

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


def _check_vision_deps() -> None:
    """Check if vision dependencies are available."""
    if not PILLOW_AVAILABLE:
        raise ImportError(
            "Vision features require the 'vision' extra. "
            "Install with: pip install fasteval[vision]"
        )


def load_image_as_base64(
    source: Union[str, Path, bytes],
    format: str = "auto",
) -> str:
    """
    Load an image and return base64-encoded string.

    Args:
        source: Image source (file path, URL, or raw bytes)
        format: Image format hint (auto, png, jpg, etc.)

    Returns:
        Base64-encoded image string with data URI prefix

    Example:
        # From file
        b64 = load_image_as_base64("chart.png")

        # From URL
        b64 = load_image_as_base64("https://example.com/image.jpg")
    """
    _check_vision_deps()

    # Handle bytes directly
    if isinstance(source, bytes):
        encoded = base64.b64encode(source).decode("utf-8")
        mime_type = _guess_mime_type(format)
        return f"data:{mime_type};base64,{encoded}"

    source_str = str(source)

    # Already base64 encoded
    if source_str.startswith("data:"):
        return source_str

    # URL - fetch and encode
    if source_str.startswith(("http://", "https://")):
        return _load_image_from_url(source_str)

    # File path
    return _load_image_from_file(source_str)


def _load_image_from_file(filepath: str) -> str:
    """Load image from file and return base64."""
    _check_vision_deps()

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {filepath}")

    with open(path, "rb") as f:
        data = f.read()

    mime_type = _guess_mime_type_from_path(path)
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _load_image_from_url(url: str) -> str:
    """Load image from URL and return base64."""
    if not HTTPX_AVAILABLE:
        raise ImportError(
            "URL image loading requires httpx. " "Install with: pip install httpx"
        )

    response = httpx.get(url, follow_redirects=True, timeout=30.0)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "image/png")
    if ";" in content_type:
        content_type = content_type.split(";")[0].strip()

    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def _guess_mime_type(format: str) -> str:
    """Guess MIME type from format string."""
    format_lower = format.lower()
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "tiff": "image/tiff",
        "auto": "image/png",
    }
    return mime_map.get(format_lower, "image/png")


def _guess_mime_type_from_path(path: Path) -> str:
    """Guess MIME type from file path."""
    suffix = path.suffix.lower().lstrip(".")
    return _guess_mime_type(suffix)


def get_image_dimensions(
    source: Union[str, Path, bytes],
) -> tuple[int, int]:
    """
    Get image dimensions (width, height).

    Args:
        source: Image source (file path, URL, or bytes)

    Returns:
        Tuple of (width, height) in pixels

    Example:
        width, height = get_image_dimensions("chart.png")
    """
    _check_vision_deps()

    if isinstance(source, bytes):
        import io

        img = Image.open(io.BytesIO(source))
        return img.size

    source_str = str(source)

    if source_str.startswith("data:"):
        # Base64 data URI
        _, data = source_str.split(",", 1)
        img_bytes = base64.b64decode(data)
        import io

        img = Image.open(io.BytesIO(img_bytes))
        return img.size

    if source_str.startswith(("http://", "https://")):
        # URL - need to download first
        if not HTTPX_AVAILABLE:
            raise ImportError("URL support requires httpx")
        response = httpx.get(source_str, follow_redirects=True, timeout=30.0)
        import io

        img = Image.open(io.BytesIO(response.content))
        return img.size

    # File path
    img = Image.open(source_str)
    return img.size


def resize_image_if_needed(
    source: Union[str, Path, bytes],
    max_dimension: int = 2048,
) -> bytes:
    """
    Resize image if it exceeds max dimension.

    Many vision APIs have size limits. This function ensures images
    fit within those limits while maintaining aspect ratio.

    Args:
        source: Image source
        max_dimension: Maximum width or height in pixels

    Returns:
        Image bytes (resized if needed, original if within limits)
    """
    _check_vision_deps()

    import io

    # Load image
    if isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
    elif str(source).startswith("data:"):
        _, data = str(source).split(",", 1)
        img_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(img_bytes))
    elif str(source).startswith(("http://", "https://")):
        if not HTTPX_AVAILABLE:
            raise ImportError("URL support requires httpx")
        response = httpx.get(str(source), follow_redirects=True, timeout=30.0)
        img = Image.open(io.BytesIO(response.content))
    else:
        img = Image.open(str(source))

    # Check if resize needed
    width, height = img.size
    if width <= max_dimension and height <= max_dimension:
        # No resize needed - return original bytes
        buffer = io.BytesIO()
        img.save(buffer, format=img.format or "PNG")
        return buffer.getvalue()

    # Calculate new dimensions
    if width > height:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))
    else:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))

    # Resize
    resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    resized.save(buffer, format=img.format or "PNG")
    return buffer.getvalue()


def normalize_image_input(
    image: Union[str, Path, bytes, "ImageInput", None],
) -> Optional[str]:
    """
    Normalize any image input to a base64 data URI.

    This is the main utility for preparing images for vision API calls.

    Args:
        image: Image in any supported format

    Returns:
        Base64 data URI string, or None if input is None

    Example:
        from fasteval.models.multimodal import ImageInput

        # All of these work:
        b64 = normalize_image_input("chart.png")
        b64 = normalize_image_input(ImageInput(source="chart.png"))
        b64 = normalize_image_input("https://example.com/image.jpg")
    """
    if image is None:
        return None

    # Handle ImageInput model
    from fasteval.models.multimodal import ImageInput

    if isinstance(image, ImageInput):
        return load_image_as_base64(image.source, image.format)

    # Handle direct inputs
    return load_image_as_base64(image)


def prepare_images_for_api(
    images: list[Union[str, Path, bytes, "ImageInput"]],
    max_dimension: int = 2048,
    max_images: Optional[int] = None,
) -> list[str]:
    """
    Prepare multiple images for API calls.

    Args:
        images: List of images in any supported format
        max_dimension: Maximum dimension for each image
        max_images: Maximum number of images to include

    Returns:
        List of base64 data URI strings
    """
    result = []

    for i, img in enumerate(images):
        if max_images is not None and i >= max_images:
            logger.warning(
                f"Truncating images: {len(images)} provided but max is {max_images}"
            )
            break

        # Unwrap ImageInput to its source
        from fasteval.models.multimodal import ImageInput

        raw = img.source if isinstance(img, ImageInput) else img

        # Resize if needed and convert to base64
        resized = resize_image_if_needed(raw, max_dimension)
        encoded = base64.b64encode(resized).decode("utf-8")
        result.append(f"data:image/png;base64,{encoded}")

    return result
