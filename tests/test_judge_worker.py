"""The judge playground runs in a warm pytest worker that imported the tests,
so it judges with whatever the tests configure (fe.set_default_provider)."""

import io
import json
import os
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from fasteval.testing import plugin
from fasteval.ui.judge_worker import JudgeWorker, JudgeWorkerError

CONFTEST = textwrap.dedent("""
    import fasteval as fe

    class ConftestJudge:
        model = "conftest-judge"
        async def invoke(self, messages):
            return '{"score": 0.9, "reasoning": "ok from conftest"}'

    fe.set_default_provider(ConftestJudge())
    """)

TESTS = textwrap.dedent("""
    import fasteval as fe

    @fe.dataset
    def questions():
        raise RuntimeError("datasets must not load in the judge worker")

    @questions
    @fe.correctness()
    def test_answers(question):
        fe.score("x", "y", input=question)
    """)

REQUEST = {
    "template": "Grade {{actual_output}}",
    "cases": [{"key": "a", "eval_input": {"actual_output": "hi"}}],
    "scoring": "continuous",
    "threshold": 0.5,
    "concurrency": 2,
}


@pytest.fixture
def project(tmp_path):
    (tmp_path / "conftest.py").write_text(CONFTEST)
    (tmp_path / "test_x.py").write_text(TESTS)
    worker = JudgeWorker(tmp_path, dict(os.environ))
    yield tmp_path, worker
    worker.close()


def test_uses_the_judge_the_tests_set_up_and_stays_warm(project):
    root, worker = project
    out = worker.run(["test_x.py"], REQUEST, [])
    assert out["judge"]["provider"] == "ConftestJudge"
    assert out["results"][0]["score"] == 0.9
    assert out["results"][0]["reasoning"] == "ok from conftest"
    pid = worker.proc.pid
    worker.run(["test_x.py"], REQUEST, [])  # a second request: same loop, same process
    worker.run([], REQUEST, ["elsewhere"])  # hand-written cases reuse the warm worker
    assert worker.proc.pid == pid


def test_restarts_when_the_tests_change(project):
    root, worker = project
    worker.run(["test_x.py"], REQUEST, [])
    pid = worker.proc.pid
    time.sleep(0.01)
    os.utime(root / "conftest.py", (time.time() + 5, time.time() + 5))
    worker.run(["test_x.py"], REQUEST, [])
    assert worker.proc.pid != pid


def test_import_errors_are_reported(project):
    root, worker = project
    (root / "test_x.py").write_text("raise ValueError('broken setup')\n")
    with pytest.raises(JudgeWorkerError, match="broken setup"):
        worker.run(["test_x.py"], REQUEST, [])
    assert worker.proc is None  # retried on the next test


# ── in-process: the plugin side (coverage can't see the worker subprocess) ──


class FakeJudge:
    model = "fake"

    async def invoke(self, messages):
        return '{"score": 0.8, "reasoning": "fine"}'


class FakeConfig:
    def __init__(self, value):
        self.value = value
        self.stash = {}

    def getoption(self, name, default=None):
        return self.value


@pytest.fixture
def fake_judge():
    from fasteval.providers.registry import clear_default_provider, set_default_provider

    set_default_provider(FakeJudge())
    yield
    clear_default_provider()


def write_request(tmp_path, name="req.json", **extra):
    path = tmp_path / name
    path.write_text(json.dumps({**REQUEST, **extra, "output": str(path) + ".out"}))
    return path


def read_output(path):
    return json.loads(Path(str(path) + ".out").read_text())


def test_answer_scores_with_the_default_provider(tmp_path, fake_judge):
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        path = write_request(tmp_path)
        plugin._answer_judge_test(str(path), [], loop)
        out = read_output(path)
        assert out["judge"]["provider"] == "FakeJudge"
        assert out["results"][0]["score"] == 0.8 and out["results"][0]["passed"]

        plugin._answer_judge_test(str(path), ["test_x.py: boom"], loop)
        assert "boom" in read_output(path)["error"]

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"output": str(bad) + ".out"}))  # no template
        plugin._answer_judge_test(str(bad), [], loop)
        assert read_output(bad)["error"].startswith("KeyError")
    finally:
        loop.close()


def test_serve_answers_each_stdin_request(tmp_path, fake_judge, monkeypatch, capsys):
    a, b = write_request(tmp_path, "a.json"), write_request(tmp_path, "b.json")
    monkeypatch.setattr(sys, "stdin", io.StringIO(f"{a}\n\n{b}\n"))
    plugin._serve_judge_tests(SimpleNamespace(config=FakeConfig("-")))
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"{plugin.JUDGE_TEST_DONE} {a}", f"{plugin.JUDGE_TEST_DONE} {b}"]
    assert read_output(a)["results"] and read_output(b)["results"]

    one = write_request(tmp_path, "one.json")
    plugin._serve_judge_tests(SimpleNamespace(config=FakeConfig(str(one))))
    assert read_output(one)["results"]


def test_hooks_skip_items_and_record_import_errors(monkeypatch):
    config = FakeConfig("-")
    collector = SimpleNamespace(config=config)
    assert plugin.pytest_pycollect_makeitem(collector, "test_x", lambda: 0) == []
    monkeypatch.setattr(plugin, "_current_config", config)
    plugin.pytest_collectreport(
        SimpleNamespace(failed=True, nodeid="test_x.py", longreprtext="boom")
    )
    plugin.pytest_collectreport(SimpleNamespace(failed=False))
    assert config.stash[plugin._JUDGE_TEST_ERRORS] == ["test_x.py: boom"]
    assert plugin._judge_test_file(FakeConfig(None)) is None
