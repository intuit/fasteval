"""Tests for fasteval.utils.text."""

from fasteval.utils.text import truncate


class TestTruncate:
    def test_empty_string(self):
        assert truncate("") == ""

    def test_short_text_unchanged(self):
        assert truncate("Hello", max_length=80) == "Hello"

    def test_long_text_truncated(self):
        result = truncate("a" * 100, max_length=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_newlines_replaced(self):
        result = truncate("line1\nline2\nline3", max_length=80)
        assert "\n" not in result
        assert result == "line1 line2 line3"

    def test_custom_ellipsis(self):
        result = truncate("a" * 100, max_length=10, ellipsis="~~")
        assert result.endswith("~~")
        assert len(result) == 10

    def test_max_length_less_than_ellipsis(self):
        result = truncate("a" * 100, max_length=2, ellipsis="...")
        assert result == ".."
        assert len(result) == 2

    def test_exact_max_length_no_truncation(self):
        text = "a" * 80
        assert truncate(text, max_length=80) == text

    def test_whitespace_stripped(self):
        result = truncate("  hello  ", max_length=80)
        assert result == "hello"
