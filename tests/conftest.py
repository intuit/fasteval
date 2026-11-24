"""
Location for your shared test fixtures
"""

import pytest


@pytest.fixture
def shared_fixture():
    """
    Placeholder for a pytest fixture that would be shared across tests.
    :return: Replace with the value your real fixture returns
    """
    pass


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """
    A tiny stand-in for the built React UI bundle.

    CI's Python test jobs don't build ui/ (a separate job does), so tests that
    render the UI or the static HTML report must use this, never the real
    fasteval/ui/static/dist.
    """
    from fasteval.ui import export, server

    dist = tmp_path / "_dist"
    dist.mkdir()
    (dist / "index.html").write_text(
        "<!doctype html><html><head><title>fasteval</title>"
        '<script type="module" crossorigin src="/static/dist/app.js"></script>'
        '<link rel="stylesheet" crossorigin href="/static/dist/app.css"></head>'
        '<body><div id="root"></div></body></html>'
    )
    (dist / "app.js").write_text("console.log('</script> safe')")
    (dist / "app.css").write_text("body{}")
    monkeypatch.setattr(export, "DIST_DIR", dist)
    monkeypatch.setattr(server, "DIST_DIR", dist)
    return dist
