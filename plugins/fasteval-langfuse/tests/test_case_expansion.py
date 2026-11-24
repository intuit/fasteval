"""In-process unit tests for per-case expansion (the subprocess e2e tests live
in test_pytest_cases.py; these cover the same code paths for coverage)."""

import inspect
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import fasteval as fe
import fasteval_langfuse.decorators as d
from fasteval_langfuse import langfuse_dataset, langfuse_traces


@pytest.fixture
def lf(monkeypatch):
    """Mock Langfuse: 3 traces (t1 fails `contains hello`), 2 dataset items."""
    traces = [
        {"id": "t0", "output": "hello a"},
        {"id": "t1", "output": "bye"},
        {"id": "t2", "output": "hello c"},
    ]
    fetcher = MagicMock()
    fetcher.fetch_and_sample.return_value = (traces, 10)
    fetcher.map_trace_to_params.side_effect = lambda t: {
        "trace_id": t["id"],
        "input": "q",
        "output": t["output"],
        "context": None,
        "metadata": {},
    }
    pushes, scores = [], []

    class Reporter:
        def __init__(self, client):
            pass

        def push_evaluation_result(self, trace_id, metric_results, aggregate_score):
            pushes.append((trace_id, aggregate_score))

        def flush(self):
            pass

    class Item:
        def __init__(self, id, metadata):
            self.id, self.input, self.expected_output, self.metadata = (
                id,
                "q",
                "hello",
                metadata,
            )

        @contextmanager
        def run(self, run_name):
            yield SimpleNamespace(
                score_trace=lambda name, value, comment=None: scores.append(
                    (self.id, name, value, run_name)
                )
            )

    client = MagicMock()
    client.fetch_dataset_raw.return_value = [
        Item("i1", {"tier": "gold"}),
        Item("i2", {}),
    ]
    monkeypatch.setattr(d, "LangfuseClient", MagicMock(return_value=client))
    monkeypatch.setattr(d, "TraceFetcher", MagicMock(return_value=fetcher))
    monkeypatch.setattr(d, "ScoreReporter", Reporter)
    monkeypatch.delenv("PYTEST_XDIST_TESTRUNUID", raising=False)
    monkeypatch.delenv("FASTEVAL_RUN_ID", raising=False)
    return SimpleNamespace(fetcher=fetcher, client=client, pushes=pushes, scores=scores)


def as_module_level(fn):
    """Real test files define tests at module level, where fe.score() finds
    the decorated function by name; mirror that for these nested helpers."""
    globals()[fn.__name__] = fn
    return fn


def make_traces_test(**kwargs):
    @fe.contains()
    @langfuse_traces(project="demo", **kwargs)
    def lf_traces_case(trace_id, input, output, my_fixture=None):
        fe.score(output, "hello", input=input)

    return as_module_level(lf_traces_case)


class TestTraceCases:
    def test_cases_and_signature(self, lf):
        fn = make_traces_test()
        cases = fn._fasteval_dataset.load()
        assert [c.id for c in cases] == ["t0", "t1", "t2"]
        assert cases[0].values[d.CASE_PARAM] == {"trace_id": "t0"}
        assert fn._fasteval_data["type"] == "langfuse_traces"
        params = list(inspect.signature(fn).parameters)
        assert params == ["trace_id", "input", "output", "my_fixture", d.CASE_PARAM]

    def test_case_mode_pushes_pass_and_fail(self, lf):
        fn = make_traces_test()
        for c in fn._fasteval_dataset.load():
            values = {
                k: v for k, v in c.values.items() if k != "context" and k != "metadata"
            }
            if c.id == "t1":
                with pytest.raises(fe.EvaluationFailedError):
                    fn(**values)
            else:
                fn(**values)
        assert [t for t, _ in lf.pushes] == ["t0", "t1", "t2"]
        assert dict(lf.pushes)["t1"] == 0.0

    def test_no_push_when_disabled(self, lf):
        fn = make_traces_test(auto_push_scores=False)
        c = fn._fasteval_dataset.load()[0]
        fn(
            trace_id="t0",
            input="q",
            output="hello",
            **{d.CASE_PARAM: c.values[d.CASE_PARAM]},
        )
        assert lf.pushes == []

    def test_sampling_stats_printed(self, lf, capsys):
        sampling = MagicMock()
        sampling.name = "RandomSamplingStrategy"
        make_traces_test(sampling=sampling)._fasteval_dataset.load()
        assert capsys.readouterr().out.strip()

    def test_async_function(self, lf):
        @fe.contains()
        @langfuse_traces(project="demo")
        async def lf_async_case(output):
            fe.score(output, "hello")

        as_module_level(lf_async_case)
        assert not inspect.iscoroutinefunction(
            lf_async_case
        )  # no pytest-asyncio needed
        lf_async_case(output="hello", **{d.CASE_PARAM: {"trace_id": "t0"}})
        # Regression: async results live in the task's context and were never pushed.
        assert lf.pushes == [("t0", 1.0)]
        with pytest.raises(fe.EvaluationFailedError):
            lf_async_case(output="nope", **{d.CASE_PARAM: {"trace_id": "t1"}})
        assert lf.pushes[-1] == ("t1", 0.0)

    def test_direct_call_runs_all_and_raises_once(self, lf):
        fn = make_traces_test()
        with pytest.raises(AssertionError, match="1 of 3 traces failed: t1"):
            fn()
        assert [t for t, _ in lf.pushes] == ["t0", "t1", "t2"]


class TestDatasetCases:
    def make(self):
        @fe.contains()
        @langfuse_dataset(name="golden", version="v1")
        def lf_dataset_case(input, expected_output, tier):
            fe.score("hello" if tier != "free" else "x", expected_output, input=input)

        return as_module_level(lf_dataset_case)

    def test_missing_metadata_is_none_and_one_run(self, lf):
        fn = self.make()
        cases = fn._fasteval_dataset.load()
        assert [c.id for c in cases] == ["i1", "i2"]
        assert cases[1].values["tier"] is None
        runs = {c.values[d.CASE_PARAM]["run_name"] for c in cases}
        assert len(runs) == 1
        for c in cases:
            fn(**{k: v for k, v in c.values.items() if k != "item_id"})
        assert {s[0] for s in lf.scores} == {"i1", "i2"}
        lf.client.flush.assert_called()

    def test_direct_call(self, lf):
        self.make()()
        assert {s[0] for s in lf.scores} == {"i1", "i2"}


class TestHelpers:
    def test_accepted_filters_unless_var_kwargs(self):
        def a(x):
            pass

        def b(x, **kw):
            pass

        assert d._accepted(a, {"x": 1, "y": 2}) == {"x": 1}
        assert d._accepted(b, {"x": 1, "y": 2}) == {"x": 1, "y": 2}

    def test_raise_failures_truncates(self):
        failures = [(f"t{i}", ValueError("x")) for i in range(12)]
        with pytest.raises(
            AssertionError, match=r"12 of 20 traces failed: .* \(\+2 more\)"
        ):
            d._raise_failures(failures, 20, "traces")
        d._raise_failures([], 3, "traces")  # no-op

    def test_run_name_shared_by_run_id(self, monkeypatch):
        monkeypatch.setenv("PYTEST_XDIST_TESTRUNUID", "abc")
        assert d._experiment_run_name("t") == "fasteval-t-abc"
        monkeypatch.setenv("FASTEVAL_RUN_ID", "run1")
        assert d._experiment_run_name("t") == "fasteval-t-run1"

    def test_shared_fetch_writer_and_reader(self, monkeypatch, tmp_path):
        monkeypatch.setattr(d.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.delenv("PYTEST_XDIST_TESTRUNUID", raising=False)
        assert d._shared_across_xdist_workers({"k": 1}, lambda: [{"a": 1}]) == [
            {"a": 1}
        ]
        assert list(tmp_path.iterdir()) == []  # no files outside xdist

        monkeypatch.setenv("PYTEST_XDIST_TESTRUNUID", "run-xyz")
        first = d._shared_across_xdist_workers({"k": 1}, lambda: [{"a": 1}])
        # A second worker gets the first worker's result, not its own fetch.
        second = d._shared_across_xdist_workers({"k": 1}, lambda: [{"a": 999}])
        assert first == second == [{"a": 1}]

    def test_shared_fetch_times_out(self, monkeypatch, tmp_path):
        monkeypatch.setattr(d.tempfile, "gettempdir", lambda: str(tmp_path))
        monkeypatch.setenv("PYTEST_XDIST_TESTRUNUID", "run-stuck")
        monkeypatch.setattr(d, "SHARED_FETCH_TIMEOUT_S", 0.2)
        key = {"k": 2}
        digest = d.hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        lock = os.path.join(str(tmp_path), f"fasteval-langfuse-run-stuck-{digest}.lock")
        open(lock, "w").close()  # another worker "holds" the fetch but never finishes
        with pytest.raises(TimeoutError):
            d._shared_across_xdist_workers(key, lambda: [])
