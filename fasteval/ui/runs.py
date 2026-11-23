"""
Run history and launching, shared by ``fasteval run`` and ``fasteval ui``.

Every run gets a directory ``.fasteval/runs/<run id>/`` containing:

    meta.json      who/what started the run: the RunRequest, the pytest
                   command, the process id, origin (cli or ui)
    events*.jsonl  live event log written by the pytest plugin
    result.json    final summary, written when the run ends
    output.log     pytest console output (runs started from the UI)

Because the request is stored, any run can be re-run with the same
configuration, or re-run with only the cases that failed.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fasteval.cli.config import ProjectConfig
from fasteval.testing.events import EVENTS_FILE, RESULT_FILE

RUNS_DIR = "runs"
META_FILE = "meta.json"
LOG_FILE = "output.log"
KEEP_RUNS = 50


@dataclass
class RunRequest:
    """What to run. Mirrors the ``fasteval run`` selection flags."""

    profile: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    keyword: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    last_failed: bool = False
    exitfirst: bool = False
    include_all: bool = False
    workers: Optional[str] = None
    judges: List[str] = field(default_factory=list)  # METRIC=JUDGE swaps
    min_score: Optional[float] = None
    min_pass_rate: Optional[float] = None
    verbose: int = 0
    quiet: bool = False
    no_summary: bool = False
    json_path: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)  # passed to pytest

    @classmethod
    def from_args(cls, args: Any, extra: List[str]) -> "RunRequest":
        """From parsed ``fasteval run`` arguments."""
        return cls(
            profile=args.profile,
            tags=list(args.tag or []),
            keyword=args.keyword,
            paths=list(args.paths or []),
            last_failed=bool(args.last_failed),
            exitfirst=bool(args.exitfirst),
            include_all=bool(args.all),
            workers=args.workers,
            judges=list(args.judge or []),
            min_score=args.min_score,
            min_pass_rate=args.min_pass_rate,
            verbose=int(args.verbose or 0),
            quiet=bool(args.quiet),
            no_summary=bool(args.no_summary),
            json_path=args.json,
            extra_args=list(extra),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunRequest":
        def str_list(key: str) -> List[str]:
            value = data.get(key) or []
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list) or not all(
                isinstance(v, str) for v in value
            ):
                raise ValueError(f"'{key}' must be a list of strings")
            return [v for v in value if v.strip()]

        def opt_float(key: str) -> Optional[float]:
            value = data.get(key)
            if value is None or value == "":
                return None
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"'{key}' must be a number")
            return float(value)

        def opt_str(key: str) -> Optional[str]:
            value = data.get(key)
            if value is None or value == "":
                return None
            if not isinstance(value, (str, int)):
                raise ValueError(f"'{key}' must be a string")
            return str(value)

        return cls(
            profile=opt_str("profile"),
            tags=str_list("tags"),
            keyword=opt_str("keyword"),
            paths=str_list("paths"),
            last_failed=bool(data.get("last_failed", False)),
            exitfirst=bool(data.get("exitfirst", False)),
            include_all=bool(data.get("include_all", False)),
            workers=opt_str("workers"),
            judges=str_list("judges"),
            min_score=opt_float("min_score"),
            min_pass_rate=opt_float("min_pass_rate"),
            verbose=int(data.get("verbose") or 0),
            quiet=bool(data.get("quiet", False)),
            no_summary=bool(data.get("no_summary", False)),
            json_path=opt_str("json_path"),
            extra_args=str_list("extra_args"),
        )

    def to_command(self) -> str:
        """The ``fasteval run ...`` line that reproduces this request."""
        import shlex

        args = self.to_cli_args()
        if self.extra_args:
            args += self.extra_args
        return shlex.join(["fasteval", "run", *args])

    def to_cli_args(self) -> List[str]:
        """Equivalent ``fasteval run`` arguments."""
        args: List[str] = list(self.paths)
        if self.profile:
            args += ["-p", self.profile]
        for tag in self.tags:
            args += ["-t", tag]
        if self.keyword:
            args += ["-k", self.keyword]
        if self.last_failed:
            args.append("--lf")
        if self.exitfirst:
            args.append("-x")
        if self.include_all:
            args.append("--all")
        if self.workers:
            args += ["-n", self.workers]
        for judge in self.judges:
            args += ["--judge", judge]
        if self.min_score is not None:
            args += ["--min-score", str(self.min_score)]
        if self.min_pass_rate is not None:
            args += ["--min-pass-rate", str(self.min_pass_rate)]
        if self.verbose:
            args.append("-" + "v" * self.verbose)
        if self.quiet:
            args.append("-q")
        if self.no_summary:
            args.append("--no-summary")
        if self.json_path:
            args += ["--json", self.json_path]
        return args


def runs_root(config: ProjectConfig) -> Path:
    return config.root / ".fasteval" / RUNS_DIR


def new_run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(2)}"


def run_dir(config: ProjectConfig, run_id: str) -> Optional[Path]:
    """Directory for ``run_id``, or None if it doesn't exist or is invalid."""
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return None
    path = runs_root(config) / run_id
    return path if path.is_dir() else None


def create_run_dir(config: ProjectConfig) -> tuple[str, Path]:
    root = runs_root(config)
    root.mkdir(parents=True, exist_ok=True)
    gitignore = config.root / ".fasteval" / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# created by fasteval\n*\n", encoding="utf-8")
    run_id = new_run_id()
    path = root / run_id
    path.mkdir()
    return run_id, path


def write_meta(path: Path, meta: Dict[str, Any]) -> None:
    tmp = path / (META_FILE + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    tmp.replace(path / META_FILE)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def make_meta(
    run_id: str,
    request: RunRequest,
    command: str,
    origin: str,
    pid: int,
    profile: str,
) -> Dict[str, Any]:
    return {
        "id": run_id,
        "origin": origin,
        "profile": profile,
        "request": asdict(request),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": pid,
    }


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def run_status(path: Path, meta: Dict[str, Any], result: Dict[str, Any]) -> str:
    if result:
        return "passed" if result.get("exit_status") == 0 else "failed"
    if pid_alive(meta.get("pid")):
        return "running"
    return "interrupted"


def summarize_run(path: Path) -> Dict[str, Any]:
    """Compact record for the runs list."""
    meta = read_json(path / META_FILE)
    result = read_json(path / RESULT_FILE)
    summary = result.get("summary", {})
    entry: Dict[str, Any] = {
        "id": path.name,
        "origin": meta.get("origin"),
        "profile": meta.get("profile"),
        "command": meta.get("command"),
        "started_at": meta.get("started_at"),
        "finished_at": result.get("finished_at"),
        "status": run_status(path, meta, result),
        "exit_status": result.get("exit_status"),
        "gate_failures": result.get("gate_failures", []),
        "request": meta.get("request", {}),
    }
    if summary:
        entry.update(
            total=summary.get("total_tests", 0),
            passed=summary.get("passed_tests", 0),
            failed=summary.get("failed_tests", 0),
            pass_rate=summary.get("pass_rate", 0.0),
            avg_score=summary.get("avg_aggregate_score", 0.0),
        )
    return entry


def list_runs(config: ProjectConfig) -> List[Dict[str, Any]]:
    root = runs_root(config)
    if not root.is_dir():
        return []
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    return [summarize_run(p) for p in dirs]


def failed_nodeids(path: Path) -> List[str]:
    """Node ids whose final outcome in this run was failed or error."""
    outcomes: Dict[str, str] = {}
    events = path / EVENTS_FILE
    if not events.exists():
        return []
    with events.open(encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "test_end":
                outcome = event.get("outcome")
                # A teardown error overrides a passed call; never downgrade.
                if outcomes.get(event["nodeid"]) not in ("failed", "error"):
                    outcomes[event["nodeid"]] = outcome
    return [n for n, o in outcomes.items() if o in ("failed", "error")]


def prune_runs(config: ProjectConfig, keep: int = KEEP_RUNS) -> None:
    root = runs_root(config)
    if not root.is_dir():
        return
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), reverse=True)
    for old in dirs[keep:]:
        meta = read_json(old / META_FILE)
        if not (old / RESULT_FILE).exists() and pid_alive(meta.get("pid")):
            continue  # still running
        shutil.rmtree(old, ignore_errors=True)


def launch(
    config: ProjectConfig,
    request: RunRequest,
    *,
    origin: str,
    capture_output: bool,
    force_non_interactive: bool = False,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> tuple[str, Path, "subprocess.Popen[bytes]", Dict[str, Any]]:
    """
    Start a run in a subprocess and record it.

    Returns (run id, run dir, process, meta). With ``capture_output`` the
    pytest console goes to ``output.log`` instead of the terminal.
    """
    from fasteval.cli.main import build_parser, build_run_plan

    args, unknown = build_parser().parse_known_args(
        ["run", *request.to_cli_args(), *(["--ci"] if force_non_interactive else [])]
    )
    profile = config.resolve(args.profile)
    run_id, path = create_run_dir(config)
    plan = build_run_plan(config, profile, args, [*unknown, *request.extra_args])
    command = plan.display  # underlying pytest call, minus internal plumbing
    plan.pytest_args.append(f"--fe-events={path}")
    env = {**os.environ, **plan.env, "FASTEVAL_RUN_ID": run_id}

    if capture_output:
        log = (path / LOG_FILE).open("wb")
        proc = subprocess.Popen(
            plan.argv,
            cwd=config.root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        log.close()
        pid = proc.pid
    else:
        proc = subprocess.Popen(plan.argv, env=env)
        pid = os.getpid()

    meta = make_meta(run_id, request, command, origin, pid, profile.name)
    meta["fasteval_command"] = request.to_command()
    meta["workers"] = plan.workers
    meta["gates"] = {"min_score": plan.min_score, "min_pass_rate": plan.min_pass_rate}
    meta.update(extra_meta or {})
    write_meta(path, meta)
    prune_runs(config)
    return run_id, path, proc, meta
