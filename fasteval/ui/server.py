"""
Local web UI for fasteval: live runs, run history, eval catalog, judge
playground. Standard library only (http.server + Server-Sent Events).

Security model: binds to 127.0.0.1 by default, rejects requests whose Host
header isn't this server (DNS-rebinding), and requires a per-process token
(embedded in the page) on every state-changing or LLM-calling request, so
other web pages open in the browser can't drive it.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import secrets
import signal
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

from fasteval.cli.config import ConfigError, ProjectConfig, find_config
from fasteval.testing.events import RESULT_FILE
from fasteval.ui import runs as runs_mod

DIST_DIR = Path(__file__).parent / "static" / "dist"
MAX_BODY = 5 * 1024 * 1024
TOKEN_HEADER = "X-Fasteval-Token"


class UIApp:
    """State shared by request handlers: project config, token, child runs."""

    def __init__(
        self, config: ProjectConfig, host: str = "127.0.0.1", port: int = 8484
    ):
        self.config = config
        self.host = host
        self.requested_port = port
        self.token = secrets.token_urlsafe(24)
        self.procs: Dict[str, "subprocess.Popen[bytes]"] = {}
        self.lock = threading.Lock()
        self.httpd: Optional[ThreadingHTTPServer] = None

    # ── lifecycle ──

    def bind(self) -> str:
        handler = type("Handler", (_Handler,), {"app": self})
        try:
            self.httpd = ThreadingHTTPServer((self.host, self.requested_port), handler)
        except OSError:
            # Port busy: take any free one rather than failing.
            self.httpd = ThreadingHTTPServer((self.host, 0), handler)
        self.httpd.daemon_threads = True
        return self.url

    @property
    def port(self) -> int:
        assert self.httpd is not None
        return int(self.httpd.server_address[1])

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "localhost") else self.host
        return f"http://{host}:{self.port}/"

    def serve_in_thread(self) -> threading.Thread:
        assert self.httpd is not None
        thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        thread.start()
        return thread

    def open_browser_when_ready(
        self, url: Optional[str] = None, timeout: float = 10.0
    ) -> None:
        """
        Open ``url`` (default: the UI) in the default browser once the server
        answers requests, so the page never loads before the server is up.
        Runs in the background; never raises.
        """
        target = url or self.url

        def wait_then_open() -> None:
            import urllib.request
            import webbrowser

            deadline = time.monotonic() + timeout
            probe = f"http://127.0.0.1:{self.port}/api/session"
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(probe, timeout=1) as res:
                        if res.status == 200:
                            break
                except OSError:
                    time.sleep(0.1)
            try:
                webbrowser.open(target)
            except Exception:  # pragma: no cover - no browser available
                pass

        threading.Thread(target=wait_then_open, daemon=True).start()

    def shutdown(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()

    # ── runs ──

    def start_run(
        self, request: runs_mod.RunRequest, rerun_of: Optional[str] = None
    ) -> Dict[str, Any]:
        with self.lock:
            active = [rid for rid, p in self.procs.items() if p.poll() is None]
            if active:
                raise RuntimeError(f"Run {active[0]} is still in progress")
            run_id, path, proc, meta = runs_mod.launch(
                self.config,
                request,
                origin="ui",
                capture_output=True,
                force_non_interactive=True,
                extra_meta={"rerun_of": rerun_of} if rerun_of else None,
            )
            self.procs[run_id] = proc
        threading.Thread(target=proc.wait, daemon=True).start()  # reap, avoid zombies
        return meta

    def stop_run(self, run_id: str) -> bool:
        proc = self.procs.get(run_id)
        if proc is None or proc.poll() is not None:
            return False
        path = runs_mod.run_dir(self.config, run_id)
        if path is not None:
            meta = runs_mod.read_json(path / runs_mod.META_FILE)
            meta["stopped"] = True
            runs_mod.write_meta(path, meta)
        # SIGINT lets pytest finish the session and write its summary.
        proc.send_signal(signal.SIGINT)

        def escalate() -> None:
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.terminate()

        threading.Thread(target=escalate, daemon=True).start()
        return True

    # ── judges ──

    def judge_client(self, model: Optional[str], temperature: Optional[float]) -> Any:
        """Client for playground calls: explicit model > judge_client > default."""
        from fasteval.judges import JudgeSpec, client_for_spec

        if model:
            return client_for_spec(
                JudgeSpec(
                    name="playground", prompt="x", model=model, temperature=temperature
                )
            )
        if self.config.judge_client:
            from fasteval.judges import JudgeError, load_judge_client

            try:
                return load_judge_client(self.config.judge_client, self.config.root)
            except JudgeError as e:
                raise HTTPError(HTTPStatus.BAD_REQUEST, str(e))
        from fasteval.providers.registry import get_default_provider

        return get_default_provider()


# ── request handling ─────────────────────────────────────────────────────────


class HTTPError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _Handler(BaseHTTPRequestHandler):
    app: UIApp
    _raw_body: bytes = b""
    server_version = "fasteval-ui"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # quiet
        pass

    # ── plumbing ──

    def _check_host(self) -> None:
        host = (self.headers.get("Host") or "").lower()
        port = self.app.port
        allowed = {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}
        if self.app.host not in ("127.0.0.1", "localhost"):
            allowed.add(f"{self.app.host}:{port}")
        if host not in allowed:
            raise HTTPError(HTTPStatus.FORBIDDEN, "Host not allowed")

    def _check_token(self) -> None:
        if not secrets.compare_digest(
            self.headers.get(TOKEN_HEADER, ""), self.app.token
        ):
            raise HTTPError(HTTPStatus.FORBIDDEN, "Missing or invalid token")

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, status: int = 200) -> None:
        self._send(status, json.dumps(data, default=str).encode(), "application/json")

    def _body(self) -> Dict[str, Any]:
        raw = self._raw_body or b"{}"
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            raise HTTPError(HTTPStatus.BAD_REQUEST, "Body must be JSON")
        if not isinstance(data, dict):
            raise HTTPError(HTTPStatus.BAD_REQUEST, "Body must be a JSON object")
        return data

    def _dispatch(self, method: str) -> None:
        try:
            self._check_host()
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            if method == "POST":
                # Always drain the body, even for handlers that ignore it:
                # leftover bytes would corrupt the next request on this
                # keep-alive connection (seen as "501 Unsupported method").
                length = int(self.headers.get("Content-Length") or 0)
                if length > MAX_BODY:
                    self.close_connection = True
                    raise HTTPError(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Body too large"
                    )
                self._raw_body = self.rfile.read(length) if length else b""
                self._check_token()
            handler = ROUTES.get((method, _route_key(parts)))
            if handler is None:
                raise HTTPError(HTTPStatus.NOT_FOUND, "Not found")
            handler(self, parts, query)
        except HTTPError as e:
            self._json({"error": e.message}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ConfigError, ValueError, RuntimeError) as e:
            self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
        except Exception as e:  # pragma: no cover - surfaced to the UI
            self._json(
                {"error": f"{type(e).__name__}: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    # ── pages ──

    def page_index(self, parts: List[str], query: Dict[str, str]) -> None:
        from fasteval.ui.export import BUILD_HINT, bundle_available

        if not bundle_available():
            body = f"<!doctype html><title>fasteval</title><pre>{BUILD_HINT}</pre>"
            self._send(503, body.encode(), "text/html; charset=utf-8")
            return
        html = (DIST_DIR / "index.html").read_bytes()
        self._send(200, html, "text/html; charset=utf-8")

    def page_static(self, parts: List[str], query: Dict[str, str]) -> None:
        name = parts[2]
        # Only plain file names that exist in the bundle (no traversal).
        allowed = (
            {p.name for p in DIST_DIR.glob("*") if p.is_file()}
            if DIST_DIR.is_dir()
            else set()
        )
        if name not in allowed:
            raise HTTPError(HTTPStatus.NOT_FOUND, "Not found")
        ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype.endswith("javascript"):
            ctype += "; charset=utf-8"
        self._send(200, (DIST_DIR / name).read_bytes(), ctype)

    def api_session(self, parts: List[str], query: Dict[str, str]) -> None:
        # Same-origin only: other sites can't read this (no CORS headers),
        # and the Host check above blocks DNS rebinding.
        self._json({"token": self.app.token})

    # ── API: project / evals ──

    def api_project(self, parts: List[str], query: Dict[str, str]) -> None:
        from fasteval.ui.export import project_info

        self.app.config = find_config(self.app.config.root)
        self._json(project_info(self.app.config))

    def api_evals(self, parts: List[str], query: Dict[str, str]) -> None:
        from fasteval.cli.main import build_parser, collect_inventory

        argv = ["list"]
        if query.get("profile"):
            argv += ["-p", query["profile"]]
        if query.get("tags"):
            argv += ["-t", query["tags"]]
        if query.get("keyword"):
            argv += ["-k", query["keyword"]]
        if query.get("all") == "1":
            argv.append("--all")
        args = build_parser().parse_args(argv)
        config = self.app.config
        inventory, output = collect_inventory(
            config, config.resolve(args.profile), args
        )
        if inventory is None:
            raise HTTPError(
                HTTPStatus.BAD_REQUEST, "Collection failed:\n" + output[-4000:]
            )
        self._json({"items": inventory["items"]})

    # ── API: runs ──

    def api_runs(self, parts: List[str], query: Dict[str, str]) -> None:
        self._json({"runs": runs_mod.list_runs(self.app.config)})

    def _run_path(self, run_id: str) -> Path:
        path = runs_mod.run_dir(self.app.config, run_id)
        if path is None:
            raise HTTPError(HTTPStatus.NOT_FOUND, f"Run {run_id!r} not found")
        return path

    def api_run(self, parts: List[str], query: Dict[str, str]) -> None:
        path = self._run_path(parts[2])
        entry = runs_mod.summarize_run(path)
        entry["meta"] = runs_mod.read_json(path / runs_mod.META_FILE)
        entry["has_log"] = (path / runs_mod.LOG_FILE).exists()
        self._json(entry)

    def api_run_log(self, parts: List[str], query: Dict[str, str]) -> None:
        path = self._run_path(parts[2]) / runs_mod.LOG_FILE
        text = (
            path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        )
        self._send(200, text[-500_000:].encode(), "text/plain; charset=utf-8")

    def api_run_report(self, parts: List[str], query: Dict[str, str]) -> None:
        from fasteval.ui.export import UIBundleMissing, build_report

        path = self._run_path(parts[2])
        try:
            html = build_report(self.app.config, path)
        except UIBundleMissing as e:
            raise HTTPError(HTTPStatus.SERVICE_UNAVAILABLE, str(e))
        self._send(
            200,
            html.encode(),
            "text/html; charset=utf-8",
            {
                "Content-Disposition": f'attachment; filename="fasteval-{path.name}.html"'
            },
        )

    def api_run_stream(self, parts: List[str], query: Dict[str, str]) -> None:
        """SSE: replay the run's events, then follow live until it ends."""
        path = self._run_path(parts[2])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        offsets: Dict[str, int] = {}
        last_beat = time.monotonic()
        while True:
            batch = _read_new_events(path, offsets)
            for _, line in batch:
                self.wfile.write(b"data: " + line + b"\n\n")
            if batch:
                self.wfile.flush()
            meta = runs_mod.read_json(path / runs_mod.META_FILE)
            result = runs_mod.read_json(path / RESULT_FILE)
            status = runs_mod.run_status(path, meta, result)
            if status != "running" and not batch:
                # One last drain for lines flushed just before exit.
                for _, line in _read_new_events(path, offsets):
                    self.wfile.write(b"data: " + line + b"\n\n")
                payload = json.dumps(
                    {"status": status, "stopped": bool(meta.get("stopped"))}
                )
                self.wfile.write(f"event: end\ndata: {payload}\n\n".encode())
                self.wfile.flush()
                return
            if time.monotonic() - last_beat > 10:
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
                last_beat = time.monotonic()
            time.sleep(0.2)

    def api_start_run(self, parts: List[str], query: Dict[str, str]) -> None:
        request = runs_mod.RunRequest.from_dict(self._body())
        meta = self.app.start_run(request)
        self._json(meta, HTTPStatus.CREATED)

    def api_rerun(self, parts: List[str], query: Dict[str, str]) -> None:
        path = self._run_path(parts[2])
        body = self._body()
        meta = runs_mod.read_json(path / runs_mod.META_FILE)
        request = runs_mod.RunRequest.from_dict(meta.get("request") or {})
        request.last_failed = False
        if body.get("failed_only"):
            failed = runs_mod.failed_nodeids(path)
            if not failed:
                raise HTTPError(HTTPStatus.BAD_REQUEST, "That run has no failed evals")
            request.paths = failed
        if "judges" in body:
            request.judges = runs_mod.RunRequest.from_dict(
                {"judges": body["judges"]}
            ).judges
        meta = self.app.start_run(request, rerun_of=path.name)
        self._json(meta, HTTPStatus.CREATED)

    def api_stop(self, parts: List[str], query: Dict[str, str]) -> None:
        if not self.app.stop_run(parts[2]):
            raise HTTPError(
                HTTPStatus.CONFLICT,
                "Only runs started from this UI and still running can be stopped",
            )
        self._json({"stopping": True})

    # ── API: judges / playground ──

    def api_judges(self, parts: List[str], query: Dict[str, str]) -> None:
        from dataclasses import asdict

        from fasteval.judges import judges_dir, list_judges

        root = self.app.config.root
        self._json(
            {
                "dir": _display(self.app.config, str(judges_dir(root))),
                "judges": [asdict(j) for j in list_judges(root)],
            }
        )

    def api_judge_template(self, parts: List[str], query: Dict[str, str]) -> None:
        from fasteval.judges import (
            infer_template,
            template_for_metric,
            template_variables,
        )

        body = self._body()
        template = None
        method = "metric"
        if body.get("metric_type"):
            template = template_for_metric(
                str(body["metric_type"]),
                str(body.get("metric_name") or body["metric_type"]),
                _number(body, "threshold", 0.5, 0.0, 1.0),
                body.get("metric_config") or {},
            )
        if template is None and body.get("rendered_prompt"):
            template = infer_template(
                str(body["rendered_prompt"]), body.get("eval_input") or {}
            )
            method = "inferred"
        if template is None:
            raise HTTPError(
                HTTPStatus.BAD_REQUEST, "No prompt template available for this metric"
            )
        self._json(
            {
                "template": template,
                "method": method,
                "variables": template_variables(template),
            }
        )

    def api_judge_test(self, parts: List[str], query: Dict[str, str]) -> None:
        body = self._body()
        template = body.get("template")
        if not isinstance(template, str) or not template.strip():
            raise HTTPError(HTTPStatus.BAD_REQUEST, "template is required")
        cases = body.get("cases") or []
        if not isinstance(cases, list) or not cases:
            raise HTTPError(HTTPStatus.BAD_REQUEST, "Add at least one case")
        if len(cases) > 100:
            raise HTTPError(HTTPStatus.BAD_REQUEST, "At most 100 cases per test")
        temperature = _number(body, "temperature", 0.0, 0.0, 2.0)
        client = self.app.judge_client(body.get("model") or None, temperature)
        results = asyncio.run(
            _run_judge_cases(
                template=template,
                client=client,
                cases=cases,
                scoring=body.get("scoring") or "continuous",
                threshold=_number(body, "threshold", 0.5, 0.0, 1.0),
                concurrency=int(_number(body, "concurrency", 4, 1, 8)),
            )
        )
        from fasteval.utils.judge_trace import describe_judge

        self._json({"judge": describe_judge(client), "results": results})

    def api_save_judge(self, parts: List[str], query: Dict[str, str]) -> None:
        from fasteval.judges import JudgeError, JudgeSpec, save_judge

        body = self._body()
        overwrite = bool(body.pop("overwrite", False))
        try:
            spec = JudgeSpec.from_dict(body)
            path = save_judge(spec, self.app.config.root, overwrite=overwrite)
        except JudgeError as e:
            raise HTTPError(HTTPStatus.BAD_REQUEST, str(e))
        self._json(
            {"name": spec.name, "path": _display(self.app.config, str(path))},
            HTTPStatus.CREATED,
        )


def _number(
    body: Dict[str, Any], key: str, default: float, lo: float, hi: float
) -> float:
    """A numeric request field within [lo, hi]; ``default`` only when absent."""
    value = body.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPError(HTTPStatus.BAD_REQUEST, f"'{key}' must be a number")
    if not lo <= float(value) <= hi:
        raise HTTPError(
            HTTPStatus.BAD_REQUEST, f"'{key}' must be between {lo:g} and {hi:g}"
        )
    return float(value)


def _display(config: ProjectConfig, path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.root))
    except ValueError:
        return path


def _read_new_events(path: Path, offsets: Dict[str, int]) -> List[Tuple[float, bytes]]:
    """Complete new lines from every events*.jsonl, ordered by timestamp."""
    batch: List[Tuple[float, bytes]] = []
    for file in sorted(path.glob("events*.jsonl")):
        offset = offsets.get(file.name, 0)
        try:
            with file.open("rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError:
            continue
        end = data.rfind(b"\n")
        if end < 0:
            continue
        offsets[file.name] = offset + end + 1
        for line in data[: end + 1].splitlines():
            if not line.strip():
                continue
            try:
                ts = float(json.loads(line).get("ts", 0))
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                continue
            batch.append((ts, line))
    batch.sort(key=lambda item: item[0])
    return batch


async def _run_judge_cases(
    *,
    template: str,
    client: Any,
    cases: List[Dict[str, Any]],
    scoring: str,
    threshold: float,
    concurrency: int,
) -> List[Dict[str, Any]]:
    from fasteval.judges import JudgeSpec, PromptJudgeMetric, render_template
    from fasteval.models.evaluation import EvalInput
    from fasteval.utils import judge_trace

    spec = JudgeSpec(
        name="playground",
        prompt=template,
        scoring="binary" if scoring == "binary" else "continuous",
        threshold=max(0.0, min(1.0, threshold)),
    )
    metric = PromptJudgeMetric(spec, name="playground", llm_client=client)
    metric.max_retries = 1
    semaphore = asyncio.Semaphore(max(1, min(concurrency, 8)))

    async def one(case: Dict[str, Any]) -> Dict[str, Any]:
        key = case.get("key")
        try:
            eval_input = EvalInput.model_validate(case.get("eval_input") or {})
        except Exception as e:
            return {"key": key, "error": f"Invalid eval_input: {e}"}
        async with semaphore:
            start = time.perf_counter()
            with judge_trace.capture() as calls:
                result = await metric.evaluate(eval_input)
        call = calls[-1] if calls else {}
        return {
            "key": key,
            "prompt": render_template(template, eval_input),
            "response": call.get("response"),
            "error": call.get("error") or result.details.get("error"),
            "score": result.score,
            "passed": result.passed,
            "reasoning": result.reasoning,
            "latency_ms": call.get(
                "latency_ms", round((time.perf_counter() - start) * 1000, 1)
            ),
        }

    return list(await asyncio.gather(*(one(c) for c in cases)))


def _route_key(parts: List[str]) -> Tuple[str, ...]:
    """Normalize a path into a route key; ids become '*'."""
    if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
        return ("api", "runs", "*", *parts[3:])
    if len(parts) == 3 and parts[:2] == ["static", "dist"]:
        return ("static", "dist", "*")
    return tuple(parts)


ROUTES: Dict[Tuple[str, Tuple[str, ...]], Any] = {
    ("GET", ()): _Handler.page_index,
    ("GET", ("static", "dist", "*")): _Handler.page_static,
    ("GET", ("api", "session")): _Handler.api_session,
    ("GET", ("api", "project")): _Handler.api_project,
    ("GET", ("api", "evals")): _Handler.api_evals,
    ("GET", ("api", "runs")): _Handler.api_runs,
    ("GET", ("api", "runs", "*")): _Handler.api_run,
    ("GET", ("api", "runs", "*", "log")): _Handler.api_run_log,
    ("GET", ("api", "runs", "*", "report.html")): _Handler.api_run_report,
    ("GET", ("api", "runs", "*", "stream")): _Handler.api_run_stream,
    ("GET", ("api", "judges")): _Handler.api_judges,
    ("POST", ("api", "runs")): _Handler.api_start_run,
    ("POST", ("api", "runs", "*", "rerun")): _Handler.api_rerun,
    ("POST", ("api", "runs", "*", "stop")): _Handler.api_stop,
    ("POST", ("api", "judges")): _Handler.api_save_judge,
    ("POST", ("api", "judges", "template")): _Handler.api_judge_template,
    ("POST", ("api", "judges", "test")): _Handler.api_judge_test,
}
