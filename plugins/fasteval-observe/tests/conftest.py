"""Pytest configuration for fasteval-observe tests."""

import pytest


@pytest.fixture(autouse=True)
def reset_context():
    """Reset context variables after each test."""
    from fasteval_observe.decorator import set_span_id, set_trace_id

    yield

    # Reset context vars
    set_trace_id(None)
    set_span_id(None)
