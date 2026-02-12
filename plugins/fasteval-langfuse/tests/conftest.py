"""Pytest configuration and fixtures for fasteval-langfuse tests."""

import pytest


@pytest.fixture
def sample_traces():
    """Sample traces for testing."""
    return [
        {
            "id": "trace-1",
            "timestamp": "2026-02-06T10:00:00Z",
            "name": "query",
            "input": "What is Python?",
            "output": "Python is a programming language",
            "metadata": {"user_type": "free"},
            "scores": [{"name": "user_rating", "value": 4.5}],
        },
        {
            "id": "trace-2",
            "timestamp": "2026-02-06T10:01:00Z",
            "name": "query",
            "input": "What is Java?",
            "output": "Java is a programming language",
            "metadata": {"user_type": "paid"},
            "scores": [{"name": "user_rating", "value": 2.0}],
        },
        {
            "id": "trace-3",
            "timestamp": "2026-02-06T10:02:00Z",
            "name": "query",
            "input": "What is RAG?",
            "output": "RAG is Retrieval Augmented Generation",
            "metadata": {
                "user_type": "free",
                "context": ["RAG combines retrieval with generation"],
            },
            "scores": [{"name": "user_rating", "value": 5.0}],
        },
    ]
