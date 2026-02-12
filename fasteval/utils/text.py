"""Text manipulation utilities."""


def truncate(text: str, max_length: int = 80, ellipsis: str = "...") -> str:
    """
    Truncate text with ellipsis if it exceeds max_length.

    Also normalizes whitespace by replacing newlines with spaces.

    Args:
        text: The text to truncate
        max_length: Maximum length including ellipsis (default: 80)
        ellipsis: String to append when truncating (default: "...")

    Returns:
        Truncated text with ellipsis if needed, or original text if short enough

    Example:
        >>> truncate("Hello world", max_length=8)
        'Hello...'
        >>> truncate("Hi", max_length=10)
        'Hi'
    """
    if not text:
        return ""

    # Normalize whitespace
    text = text.replace("\n", " ").strip()

    if len(text) <= max_length:
        return text

    # Account for ellipsis length
    truncate_at = max_length - len(ellipsis)
    if truncate_at <= 0:
        return ellipsis[:max_length]

    return text[:truncate_at] + ellipsis
