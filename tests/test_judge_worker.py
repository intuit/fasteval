"""The judge playground runs in a warm pytest worker that imported the tests,
so it judges with whatever the tests configure (fe.set_default_provider)."""

import os
import textwrap
import time

import pytest

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
