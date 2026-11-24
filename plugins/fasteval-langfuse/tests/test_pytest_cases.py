"""End-to-end: @langfuse_traces / @langfuse_dataset under real pytest runs.

Langfuse is mocked in a generated conftest; every push / score is logged to a
file so the tests can assert what reached "Langfuse".
"""

import importlib.util
import json
import textwrap

import pytest

pytest_plugins = ["pytester"]

CONFTEST = textwrap.dedent("""
    import json, os, random
    from contextlib import contextmanager
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import fasteval_langfuse.decorators as d

    LOG = os.path.join(os.path.dirname(__file__), "langfuse.log")
    RANDOM_FETCH = os.environ.get("LF_RANDOM_FETCH") == "1"
    POOL = [{"id": f"t{i}", "input": f"q{i}", "output": "hello world" if i % 3 else "bye"}
            for i in range(12)]

    def log(**row):
        with open(LOG, "a") as f:
            f.write(json.dumps({**row, "pid": os.getpid()}) + "\\n")

    def fetch_and_sample(**kw):
        # RANDOM_FETCH mimics unseeded sampling / moving time windows: a
        # different set on every call (i.e. in every xdist worker).
        return (random.sample(POOL, 6) if RANDOM_FETCH else POOL[:6]), len(POOL)

    fetcher = MagicMock()
    fetcher.fetch_and_sample.side_effect = fetch_and_sample
    fetcher.map_trace_to_params.side_effect = lambda t: {
        "trace_id": t["id"], "input": t["input"], "output": t["output"],
        "context": None, "metadata": {}}

    class Reporter:
        def __init__(self, client):
            pass
        def push_evaluation_result(self, trace_id, metric_results, aggregate_score):
            log(kind="push", trace=trace_id, score=aggregate_score)
        def flush(self):
            log(kind="flush")

    class Item:
        def __init__(self, id, input, expected, metadata):
            self.id, self.input, self.expected_output, self.metadata = id, input, expected, metadata
        @contextmanager
        def run(self, run_name):
            span = SimpleNamespace(score_trace=lambda name, value, comment=None:
                                   log(kind="score", item=self.id, name=name, value=value, run=run_name))
            yield span

    ITEMS = [
        Item("i1", "q1", "hello", {"tier": "gold"}),
        Item("i2", "q2", "hello", {}),                  # no metadata: tier -> None
        Item("i3", "q3", "bye", {"tier": "free", "id": "meta-id"}),
    ]
    client = MagicMock()
    client.fetch_dataset_raw.return_value = ITEMS
    d.LangfuseClient = MagicMock(return_value=client)
    d.TraceFetcher = MagicMock(return_value=fetcher)
    d.ScoreReporter = Reporter
    """)

TESTS = textwrap.dedent("""
    import fasteval as fe
    from fasteval_langfuse import langfuse_dataset, langfuse_traces

    @fe.contains()
    @langfuse_traces(project="demo")
    def test_traces(trace_id, input, output):
        fe.score(output, "hello", input=input)

    @fe.contains()
    @langfuse_dataset(name="golden")
    def test_dataset(input, expected_output, tier):
        answer = "hello world" if tier != "free" else "nope"
        fe.score(answer, expected_output, input=input)
    """)


@pytest.fixture
def project(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_lf=TESTS)
    return pytester


def read_log(pytester):
    path = pytester.path / "langfuse.log"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_one_case_per_trace_and_item(project):
    result = project.runpytest_subprocess("--collect-only", "-q")
    lines = set(result.stdout.lines)
    for i in range(6):
        assert f"test_lf.py::test_traces[t{i}]" in lines
    for item in ("i1", "i2", "i3"):
        assert f"test_lf.py::test_dataset[{item}]" in lines


def test_failures_isolated_and_every_score_pushed(project):
    result = project.runpytest_subprocess("-k", "test_traces", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=4, failed=2, deselected=3)  # t0, t3 output "bye"
    pushes = [r for r in read_log(project) if r["kind"] == "push"]
    assert sorted(r["trace"] for r in pushes) == [f"t{i}" for i in range(6)]
    assert sorted(r["trace"] for r in pushes if r["score"] == 0) == ["t0", "t3"]
    assert sum(r["kind"] == "flush" for r in read_log(project)) == 6


def test_dataset_items_scored_in_one_run_with_varying_metadata(project):
    result = project.runpytest_subprocess(
        "-k", "test_dataset", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=2, failed=1, deselected=6)  # i3 (tier=free) fails
    scores = [r for r in read_log(project) if r["kind"] == "score"]
    assert {r["item"] for r in scores} == {"i1", "i2", "i3"}  # failing item scored too
    assert len({r["run"] for r in scores}) == 1  # one experiment run
    assert {r["name"].rsplit("_", 1)[-1] for r in scores} >= {"contains", "aggregate"}


def test_results_reach_fasteval_reports(project):
    out = project.path / "r.json"
    project.runpytest_subprocess("-p", "no:cacheprovider", f"--fe-output=json:{out}")
    names = {
        t["test_name"] for t in json.loads(out.read_text())["summary"]["test_summaries"]
    }
    assert "test_traces[t1]" in names and "test_dataset[i2]" in names


@pytest.mark.skipif(
    importlib.util.find_spec("xdist") is None, reason="needs pytest-xdist"
)
def test_xdist_workers_collect_identical_cases(project, monkeypatch):
    # Each worker's fetch would return a different random set; the shared
    # fetch must make them identical or xdist aborts ("Different tests...").
    monkeypatch.setenv("LF_RANDOM_FETCH", "1")
    result = project.runpytest_subprocess(
        "-n", "3", "-k", "test_traces", "-p", "no:cacheprovider"
    )
    outcomes = result.parseoutcomes()
    assert outcomes.get("passed", 0) + outcomes.get("failed", 0) == 6
    assert "Different tests were collected" not in result.stdout.str()
    pushes = [r for r in read_log(project) if r["kind"] == "push"]
    assert len(pushes) == 6 and len({r["trace"] for r in pushes}) == 6


def test_direct_call_evaluates_all_then_raises(project):
    """Outside pytest: every trace evaluated and pushed; one error at the end."""
    project.makepyfile(run_direct=textwrap.dedent("""
            import conftest  # installs the Langfuse mocks
            from test_lf import test_traces

            try:
                test_traces()
            except AssertionError as e:
                print("RAISED:", e)
            """))
    result = project.runpython(project.path / "run_direct.py")
    result.stdout.fnmatch_lines(["RAISED: 2 of 6 traces failed: t0, t3"])
    pushes = [r for r in read_log(project) if r["kind"] == "push"]
    assert len(pushes) == 6
