"""Tests for the UI backend: event log, run history, static export, server."""

import http.client
import json
import os
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

from fasteval.cli.config import find_config
from fasteval.testing.events import EventWriter, bounded
from fasteval.ui import export, runs, server

EVALS = textwrap.dedent("""
    import fasteval as fe

    @fe.tag("smoke")
    @fe.contains()
    def test_greets():
        fe.score("hello world", "hello", input="hi")

    @fe.exact_match()
    def test_wrong():
        fe.score("general", "billing", input="double charged")
    """)

PYPROJECT = textwrap.dedent("""
    [tool.pytest.ini_options]
    testpaths = ["evals"]

    [tool.fasteval]
    default_profile = "dev"
    judge_client = "judge_client:client"

    [tool.fasteval.profiles.dev]
    paths = ["evals"]
    min_pass_rate = 0.4
    """)

JUDGE_CLIENT = textwrap.dedent("""
    import json

    class Client:
        model = "fake-judge"

        async def invoke(self, messages):
            text = messages[-1]["content"]
            return json.dumps({"score": 1.0 if "hello" in text else 0.0, "reasoning": "checked"})

    client = Client()
    """)


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "test_demo.py").write_text(EVALS)
    (tmp_path / "judge_client.py").write_text(JUDGE_CLIENT)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    return tmp_path


def run_to_end(config, request=None, **kw):
    run_id, path, proc, meta = runs.launch(
        config,
        request or runs.RunRequest(quiet=True),
        origin=kw.pop("origin", "ui"),
        capture_output=True,
        **kw,
    )
    assert proc.wait(timeout=120) in (0, 1)
    return run_id, path, meta


# ── event log ────────────────────────────────────────────────────────────────


class TestEvents:
    def test_writer_files_and_result(self, tmp_path):
        main = EventWriter(tmp_path)
        worker = EventWriter(tmp_path, worker_id="gw0")
        main.emit("run_start", pid=1)
        worker.emit("eval", nodeid="t::x")
        main.write_result({"exit_status": 0})
        main.close()
        worker.close()
        main.close()
        assert (
            json.loads((tmp_path / "events.jsonl").read_text())["type"] == "run_start"
        )
        assert (
            json.loads((tmp_path / "events-gw0.jsonl").read_text())["worker"] == "gw0"
        )
        assert json.loads((tmp_path / "result.json").read_text())["exit_status"] == 0

    def test_bounded_truncates_nested_strings(self):
        out = bounded({"a": ["x" * 30, {"b": "y" * 5}]}, limit=10)
        assert out["a"][0].startswith("x" * 10) and "truncated" in out["a"][0]
        assert out["a"][1]["b"] == "yyyyy"


# ── run requests & history ───────────────────────────────────────────────────


class TestRunRequest:
    def test_round_trip_to_cli_and_command(self):
        req = runs.RunRequest.from_dict(
            {
                "profile": "ci",
                "tags": "smoke",
                "paths": ["a.py::t[x y]"],
                "judges": ["c=j"],
                "min_score": 0.5,
                "verbose": 2,
                "quiet": True,
                "no_summary": True,
                "json_path": "r.json",
                "workers": 2,
                "last_failed": True,
                "exitfirst": True,
                "include_all": True,
            }
        )
        assert req.tags == ["smoke"] and req.workers == "2"
        args = req.to_cli_args()
        for flag in ("--lf", "-x", "--all", "-vv", "-q", "--no-summary"):
            assert flag in args
        assert args[args.index("--judge") + 1] == "c=j"
        assert req.to_command().startswith("fasteval run 'a.py::t[x y]' -p ci -t smoke")

    @pytest.mark.parametrize(
        "data",
        [{"tags": [1]}, {"min_score": "high"}, {"min_score": True}, {"keyword": []}],
    )
    def test_rejects_bad_types(self, data):
        with pytest.raises(ValueError):
            runs.RunRequest.from_dict(data)

    def test_run_dir_rejects_traversal(self, project):
        config = find_config()
        for bad in ("", "../x", "a/b", ".hidden"):
            assert runs.run_dir(config, bad) is None


class TestRuns:
    def test_launch_records_events_meta_and_history(self, project):
        config = find_config()
        run_id, path, meta = run_to_end(
            config, runs.RunRequest(tags=["smoke"], quiet=True)
        )
        assert meta["fasteval_command"] == "fasteval run -t smoke -q"
        assert "--fe-events" not in meta["command"] and meta["origin"] == "ui"
        assert (config.root / ".fasteval" / ".gitignore").exists()
        assert "1 passed" in (path / runs.LOG_FILE).read_text()
        (entry,) = runs.list_runs(config)
        assert (
            entry["id"] == run_id
            and entry["status"] == "passed"
            and entry["total"] == 1
        )

    def test_failed_nodeids_and_status(self, project):
        config = find_config()
        _, path, _ = run_to_end(config)
        assert runs.failed_nodeids(path) == ["evals/test_demo.py::test_wrong"]
        # A failing eval fails the run even though the pass-rate gate (0.4) is met.
        assert runs.summarize_run(path)["status"] == "failed"
        assert runs.run_status(path, {"pid": 0}, {}) == "interrupted"

    def test_prune_keeps_newest(self, project):
        config = find_config()
        root = runs.runs_root(config)
        for i in range(5):
            (root / f"2026010{i}-000000-0000").mkdir(parents=True)
            (root / f"2026010{i}-000000-0000" / "result.json").write_text("{}")
        runs.prune_runs(config, keep=2)
        assert sorted(p.name for p in root.iterdir()) == [
            "20260103-000000-0000",
            "20260104-000000-0000",
        ]


# ── static export ────────────────────────────────────────────────────────────


class TestExport:
    def test_self_contained_report(self, project, bundle):
        config = find_config()
        _, path, _ = run_to_end(config)
        html = export.build_report(config, path)
        assert "window.FE_STATIC" in html and "<style>body{}</style>" in html
        assert 'src="/static/dist/app.js"' not in html and "<\\/script> safe" in html
        data = json.loads(
            html.split("window.FE_STATIC = ", 1)[1].split(";</script>", 1)[0]
        )
        assert data["run"]["id"] == path.name
        assert {e["type"] for e in data["events"]} >= {
            "run_start",
            "collected",
            "eval",
            "run_end",
        }

    def test_embedded_json_cannot_close_script(self):
        assert "</script>" not in export._script_json({"x": "</script><b>"})

    def test_missing_bundle(self, project, monkeypatch, tmp_path):
        monkeypatch.setattr(export, "DIST_DIR", tmp_path / "nothing")
        assert not export.bundle_available()
        with pytest.raises(export.UIBundleMissing, match="npm run build"):
            export.build_report(find_config(), tmp_path)


# ── HTTP server ──────────────────────────────────────────────────────────────


@pytest.fixture
def live(project, bundle):
    app = server.UIApp(find_config(), port=0)
    app.bind()
    app.serve_in_thread()
    yield app
    app.shutdown()


def call(app, method, path, body=None, token=True, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=30)
    headers = {"Host": host or f"localhost:{app.port}"}
    data = None
    if method == "POST":
        data = json.dumps(body or {})
        headers["Content-Type"] = "application/json"
        if token:
            headers["X-Fasteval-Token"] = app.token
    conn.request(method, path, body=data, headers=headers)
    res = conn.getresponse()
    raw = res.read()
    conn.close()
    try:
        return res.status, json.loads(raw)
    except json.JSONDecodeError:
        return res.status, raw.decode()


def wait_for_run(app, run_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, data = call(app, "GET", f"/api/runs/{run_id}")
        if data["status"] != "running":
            return data
        time.sleep(0.2)
    raise AssertionError("run did not finish")


class TestServer:
    def test_pages_and_static_files(self, live):
        assert call(live, "GET", "/")[0] == 200
        assert call(live, "GET", "/static/dist/app.js")[0] == 200
        assert call(live, "GET", "/static/dist/..%2Fapp.js")[0] == 404
        assert call(live, "GET", "/static/dist/secret.py")[0] == 404
        assert call(live, "GET", "/nope")[0] == 404

    def test_security(self, live):
        assert call(live, "GET", "/api/project", host="evil.example:80")[0] == 403
        assert call(live, "POST", "/api/runs", token=False)[0] == 403
        status, session = call(live, "GET", "/api/session")
        assert status == 200 and session["token"] == live.token

    def test_project_info(self, live):
        status, data = call(live, "GET", "/api/project")
        assert status == 200 and data["default_profile"] == "dev"
        assert data["judge_client"] == "judge_client:client"
        assert {p["name"] for p in data["profiles"]} == {"default", "dev"}

    def test_evals_inventory(self, live):
        status, data = call(live, "GET", "/api/evals?profile=dev&tags=smoke")
        assert status == 200 and [i["function"] for i in data["items"]] == [
            "test_greets"
        ]

    def test_start_stream_rerun_and_report(self, live):
        status, meta = call(live, "POST", "/api/runs", {"quiet": True})
        assert status == 201
        assert call(live, "POST", "/api/runs", {})[0] == 400  # one UI run at a time
        info = wait_for_run(live, meta["id"])
        assert info["status"] == "failed" and info["has_log"]  # test_wrong fails

        with urllib.request.urlopen(
            f"http://127.0.0.1:{live.port}/api/runs/{meta['id']}/stream"
        ) as res:
            body = res.read().decode()
        assert body.count("data: ") >= 6 and "event: end" in body

        status, rerun = call(
            live, "POST", f"/api/runs/{meta['id']}/rerun", {"failed_only": True}
        )
        assert status == 201 and rerun["rerun_of"] == meta["id"]
        assert rerun["request"]["paths"] == ["evals/test_demo.py::test_wrong"]
        wait_for_run(live, rerun["id"])

        status, html = call(live, "GET", f"/api/runs/{meta['id']}/report.html")
        assert status == 200 and "window.FE_STATIC" in html
        assert "passed" in call(live, "GET", f"/api/runs/{meta['id']}/log")[1]
        assert call(live, "GET", "/api/runs/missing-run")[0] == 404
        assert call(live, "POST", f"/api/runs/{meta['id']}/stop")[0] == 409

    def test_keep_alive_after_body_ignoring_post(self, live):
        """Regression: an unread POST body must not corrupt the next request."""
        conn = http.client.HTTPConnection("127.0.0.1", live.port, timeout=10)
        headers = {
            "Host": f"localhost:{live.port}",
            "X-Fasteval-Token": live.token,
            "Content-Type": "application/json",
        }
        conn.request("POST", "/api/runs/none/stop", body="{}", headers=headers)
        assert conn.getresponse().read() and True
        conn.request("GET", "/api/session", headers={"Host": f"localhost:{live.port}"})
        res = conn.getresponse()
        assert res.status == 200 and json.loads(res.read())["token"] == live.token
        conn.close()

    def test_judge_template_test_and_save(self, live, project):
        status, tpl = call(
            live,
            "POST",
            "/api/judges/template",
            {"metric_type": "correctness", "threshold": 0},
        )
        assert (
            status == 200
            and tpl["method"] == "metric"
            and "actual_output" in tpl["variables"]
        )

        cases = [
            {"key": "a", "eval_input": {"actual_output": "hello there"}},
            {"key": "b", "eval_input": {"actual_output": "bye"}},
        ]
        status, res = call(
            live,
            "POST",
            "/api/judges/test",
            {
                "template": "Out: {{actual_output}} score",
                "cases": cases,
                "threshold": 0.5,
            },
        )
        assert status == 200 and res["judge"]["model"] == "fake-judge"
        by_key = {r["key"]: r for r in res["results"]}
        assert by_key["a"]["passed"] and not by_key["b"]["passed"]
        assert by_key["a"]["prompt"] == "Out: hello there score"

        assert (
            call(
                live,
                "POST",
                "/api/judges/test",
                {"template": "x", "cases": cases, "threshold": 7},
            )[0]
            == 400
        )
        assert (
            call(live, "POST", "/api/judges/test", {"template": "x", "cases": []})[0]
            == 400
        )

        status, saved = call(
            live,
            "POST",
            "/api/judges",
            {"name": "j1", "prompt": "Out: {{actual_output}}"},
        )
        assert status == 201 and (project / "judges" / "j1.toml").exists()
        assert (
            call(live, "POST", "/api/judges", {"name": "j1", "prompt": "x"})[0] == 400
        )
        status, listed = call(live, "GET", "/api/judges")
        assert [j["name"] for j in listed["judges"]] == ["j1"]

    def test_run_with_judge_swap(self, live, project):
        call(
            live,
            "POST",
            "/api/judges",
            {"name": "j1", "prompt": "Out: {{actual_output}}", "threshold": 0.5},
        )
        status, meta = call(
            live, "POST", "/api/runs", {"judges": ["exact_match=j1"], "quiet": True}
        )
        assert status == 201
        info = wait_for_run(live, meta["id"])
        assert info["request"]["judges"] == ["exact_match=j1"]
        events = (
            Path(project) / ".fasteval" / "runs" / meta["id"] / "events.jsonl"
        ).read_text()
        assert '"prompt_judge": {"name": "j1"' in events


def test_missing_bundle_page(project, monkeypatch, tmp_path):
    monkeypatch.setattr(export, "DIST_DIR", tmp_path / "none")
    monkeypatch.setattr(server, "DIST_DIR", tmp_path / "none")
    app = server.UIApp(find_config(), port=0)
    app.bind()
    app.serve_in_thread()
    try:
        status, body = call(app, "GET", "/")
        assert status == 503 and "npm run build" in body
    finally:
        app.shutdown()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signals")
def test_stop_running_ui_run(project, bundle):
    (project / "evals" / "test_slow.py").write_text(
        "import time, fasteval as fe\n\n@fe.contains()\ndef test_slow():\n    time.sleep(30)\n    fe.score('a', 'a')\n"
    )
    app = server.UIApp(find_config(), port=0)
    app.bind()
    app.serve_in_thread()
    try:
        status, meta = call(app, "POST", "/api/runs", {"keyword": "slow"})
        assert status == 201
        time.sleep(1.5)
        assert call(app, "POST", f"/api/runs/{meta['id']}/stop")[0] == 200
        info = wait_for_run(app, meta["id"], timeout=30)
        assert info["status"] in ("failed", "interrupted") and info["meta"]["stopped"]
    finally:
        app.shutdown()


def test_explicit_default_provider_beats_judge_client(project):
    """judge_client is only the fallback; fe.set_default_provider wins."""
    (project / "judges").mkdir()
    (project / "judges" / "j1.toml").write_text(
        "prompt = 'Out: {{actual_output}}'\nthreshold = 0.5\n"
    )
    (project / "evals" / "conftest.py").write_text(
        "import json, fasteval as fe\n\n"
        "class Mine:\n    model = 'from-conftest'\n"
        "    async def invoke(self, messages):\n        return json.dumps({'score': 1.0, 'reasoning': 'mine'})\n\n"
        "fe.set_default_provider(Mine())\n"
    )
    config = find_config()
    _, path, _ = run_to_end(
        config, runs.RunRequest(judges=["exact_match=j1"], quiet=True)
    )
    events = (path / "events.jsonl").read_text()
    assert (
        '"model": "from-conftest"' in events and '"model": "fake-judge"' not in events
    )


def test_judge_test_with_a_model_calls_it_in_process(live, monkeypatch):
    """A filled-in Judge model skips the worker and calls that model directly."""
    import fasteval.judges as judges_mod

    class Direct:
        model = "typed-model"

        async def invoke(self, messages):
            return json.dumps({"score": 0.9, "reasoning": "direct"})

    seen = {}

    def fake_client_for_spec(spec, fallback=None):
        seen["model"], seen["temperature"] = spec.model, spec.temperature
        return Direct()

    monkeypatch.setattr(judges_mod, "client_for_spec", fake_client_for_spec)
    status, res = call(
        live,
        "POST",
        "/api/judges/test",
        {
            "template": "Out: {{actual_output}}",
            "model": "typed-model",
            "temperature": 0.3,
            "cases": [{"key": "a", "eval_input": {"actual_output": "hi"}}],
        },
    )
    assert status == 200 and res["judge"]["model"] == "typed-model"
    assert res["results"][0]["score"] == 0.9
    assert seen == {"model": "typed-model", "temperature": 0.3}
    assert live.judge_worker is None  # no worker started


def test_judge_test_reports_worker_errors(live, project):
    (project / "evals" / "conftest.py").write_text("raise ImportError('no sdk here')\n")
    status, res = call(
        live,
        "POST",
        "/api/judges/test",
        {
            "template": "Out: {{actual_output}}",
            "cases": [{"key": "a", "nodeid": "evals/missing.py::t", "eval_input": {}}],
        },
    )
    assert status == 502 and "no sdk here" in res["error"]
