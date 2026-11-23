"""
fasteval command line interface.

A thin, profile-driven front door to pytest. Every command translates to a
plain pytest invocation (shown before it runs), so nothing the CLI does is
magic: ``fasteval run -p ci`` is exactly the ``pytest ...`` line it prints.

Commands:
    fasteval                 project status: config, profiles, last run
    fasteval run [PATHS]     run evals (unknown flags pass through to pytest)
    fasteval list [PATHS]    inventory of evals: tags, metrics, data sources
    fasteval profiles [NAME] list profiles, or show one resolved
    fasteval report          summarize the last run; --html/--open for a report
    fasteval init            add a starter [tool.fasteval] config
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import webbrowser
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fasteval.cli import ui
from fasteval.cli.config import (
    DEFAULT_PROFILE,
    ConfigError,
    Profile,
    ProjectConfig,
    find_config,
)

STATE_DIR = ".fasteval"
LAST_RUN = "last-run.json"
LAST_RUN_META = "last-run.meta.json"

_TRUTHY = {"1", "true", "yes", "on"}


def is_ci_env() -> bool:
    return os.environ.get("CI", "").lower() in _TRUTHY


# ── argument parsing ─────────────────────────────────────────────────────────


def _add_selection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "paths",
        nargs="*",
        help="files, directories or node ids (default: the profile's paths)",
    )
    p.add_argument("-p", "--profile", help="profile from [tool.fasteval.profiles]")
    p.add_argument(
        "-t",
        "--tag",
        action="append",
        default=[],
        metavar="EXPR",
        help="select by tag (pytest marker expression); repeat to OR. "
        "Narrows the profile's own tag selection.",
    )
    p.add_argument(
        "-k", dest="keyword", metavar="EXPR", help="select by test name expression"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="include plain tests, not only fasteval evals",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fasteval",
        description="Run, browse and report on fasteval evals. Built on pytest.",
    )
    parser.add_argument("--version", action="store_true", help="show version")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    run = sub.add_parser(
        "run",
        help="run evals",
        description="Run evals. Flags fasteval doesn't know are passed to pytest.",
    )
    _add_selection_args(run)
    run.add_argument(
        "-n", "--workers", help="parallel workers via pytest-xdist (int or 'auto')"
    )
    run.add_argument(
        "--lf",
        "--last-failed",
        dest="last_failed",
        action="store_true",
        help="rerun only evals that failed last time (saves LLM calls)",
    )
    run.add_argument(
        "-x", "--exitfirst", action="store_true", help="stop on first failure"
    )
    run.add_argument("-v", "--verbose", action="count", default=0)
    run.add_argument("-q", "--quiet", action="store_true")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless",
        action="store_true",
        help="no UI server: run, write a self-contained HTML report, exit "
        "(default when $CI is set or there's no terminal)",
    )
    mode.add_argument(
        "--ui", action="store_true", help="force the live UI even without a terminal"
    )
    run.add_argument(
        "--no-open", action="store_true", help="don't open the browser in UI mode"
    )
    run.add_argument(
        "--port", type=int, default=8484, help="UI server port (default 8484)"
    )
    run.add_argument(
        "--html",
        metavar="PATH",
        help=f"static HTML report path (default: {STATE_DIR}/report.html)",
    )
    run.add_argument("--json", metavar="PATH", help="write a JSON report")
    run.add_argument("--min-score", type=float, help="fail if avg score is below")
    run.add_argument(
        "--min-pass-rate", type=float, help="fail if eval pass rate is below"
    )
    run.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: headless and no human-review prompts (implied by $CI)",
    )
    run.add_argument(
        "--no-summary", action="store_true", help="skip the fasteval summary"
    )
    run.add_argument(
        "--judge",
        action="append",
        default=[],
        metavar="METRIC=JUDGE",
        help="swap a metric for a saved prompt judge (judges/JUDGE.toml)",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="print the pytest command and exit"
    )

    ls = sub.add_parser("list", aliases=["ls"], help="list evals")
    _add_selection_args(ls)
    ls.add_argument("--tags", dest="tags_only", action="store_true", help="tag counts")
    ls.add_argument("--json", dest="as_json", action="store_true", help="JSON output")

    prof = sub.add_parser("profiles", help="list profiles or show one")
    prof.add_argument("name", nargs="?", help="profile to show, fully resolved")

    rep = sub.add_parser("report", help="summarize a run; export it as HTML")
    rep.add_argument("run_id", nargs="?", help="run id (default: the latest)")
    rep.add_argument(
        "--html",
        nargs="?",
        const="",
        metavar="PATH",
        help=f"write the self-contained HTML report (default: {STATE_DIR}/report.html)",
    )
    rep.add_argument("--open", action="store_true", help="open the HTML report")
    rep.add_argument("--list", action="store_true", help="list recorded runs")

    uip = sub.add_parser("ui", help="open the local UI: runs, evals, judges")
    uip.add_argument("--port", type=int, default=8484, help="port (default 8484)")
    uip.add_argument("--host", default="127.0.0.1", help="bind address")
    uip.add_argument("--no-open", action="store_true", help="don't open a browser")

    sub.add_parser("init", help="add a starter [tool.fasteval] config")
    return parser


# ── translating a profile into pytest arguments ──────────────────────────────


def combine_and(*exprs: Optional[str]) -> Optional[str]:
    """AND together non-empty selection expressions."""
    parts = [e.strip() for e in exprs if e and e.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " and ".join(f"({p})" for p in parts)


def combine_or(exprs: Sequence[str]) -> Optional[str]:
    """OR together non-empty selection expressions."""
    parts = [e.strip() for e in exprs if e and e.strip()]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return " or ".join(f"({p})" for p in parts)


@dataclass
class Plan:
    """Everything needed to launch (and describe) one pytest invocation."""

    profile: Profile
    pytest_args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    marker_expr: Optional[str] = None
    keyword_expr: Optional[str] = None
    paths: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    workers: Optional[str] = None
    review_prompts: bool = True
    min_score: Optional[float] = None
    min_pass_rate: Optional[float] = None
    judges: List[str] = field(default_factory=list)

    @property
    def argv(self) -> List[str]:
        return [sys.executable, "-m", "pytest", *self.pytest_args]

    @property
    def display(self) -> str:
        return "pytest " + shlex.join(self.pytest_args)


def _display_path(path: str) -> str:
    try:
        rel = os.path.relpath(path)
    except ValueError:  # pragma: no cover - different drive on Windows
        return path
    return path if rel.startswith(".." + os.sep + "..") else rel


def build_selection(
    profile: Profile, args: argparse.Namespace
) -> tuple[List[str], Optional[str], Optional[str], List[str]]:
    """Selection args shared by ``run`` and ``list``."""
    paths = list(args.paths) or [_display_path(p) for p in profile.paths]
    marker_expr = combine_and(profile.tags, combine_or(args.tag))
    keyword_expr = combine_and(profile.keyword, args.keyword)
    out: List[str] = list(paths)
    if not args.all:
        out.append("--fe-only")
    if marker_expr:
        out += ["-m", marker_expr]
    if keyword_expr:
        out += ["-k", keyword_expr]
    return out, marker_expr, keyword_expr, paths


def build_run_plan(
    config: ProjectConfig,
    profile: Profile,
    args: argparse.Namespace,
    passthrough: Sequence[str] = (),
) -> Plan:
    pytest_args, marker_expr, keyword_expr, paths = build_selection(profile, args)

    workers = args.workers if args.workers is not None else profile.workers
    workers_str = None if workers in (None, "", 0, "0") else str(workers)
    if workers_str:
        pytest_args += ["-n", workers_str]

    if args.last_failed:
        pytest_args.append("--lf")
    if args.exitfirst:
        pytest_args.append("-x")
    if args.verbose:
        pytest_args.append("-" + "v" * args.verbose)
    if args.quiet:
        pytest_args.append("-q")

    review_prompts = not (args.ci or is_ci_env() or profile.review_prompts is False)
    if not review_prompts:
        pytest_args.append("--no-interactive")

    if not args.no_summary:
        pytest_args.append("--fe-summary")

    outputs = list(profile.outputs)
    if getattr(args, "json", None):
        outputs.append(f"json:{args.json}")
    for spec in outputs:
        pytest_args.append(f"--fe-output={spec}")
    # Always keep the last run around for `fasteval report`.
    pytest_args.append(f"--fe-output=json:{_display_path(str(_last_run_path(config)))}")

    min_score = args.min_score if args.min_score is not None else profile.min_score
    min_pass_rate = (
        args.min_pass_rate if args.min_pass_rate is not None else profile.min_pass_rate
    )
    if min_score is not None:
        pytest_args.append(f"--fe-min-score={min_score}")
    if min_pass_rate is not None:
        pytest_args.append(f"--fe-min-pass-rate={min_pass_rate}")

    judges = list(getattr(args, "judge", []) or [])
    for spec in judges:
        pytest_args.append(f"--fe-judge={spec}")

    pytest_args += profile.pytest_args
    pytest_args += list(passthrough)

    env = dict(profile.env)
    env["FASTEVAL_PROFILE"] = profile.name

    return Plan(
        profile=profile,
        pytest_args=pytest_args,
        env=env,
        marker_expr=marker_expr,
        keyword_expr=keyword_expr,
        paths=paths,
        outputs=outputs,
        workers=workers_str,
        review_prompts=review_prompts,
        min_score=min_score,
        min_pass_rate=min_pass_rate,
        judges=judges,
    )


# ── state (.fasteval/) ───────────────────────────────────────────────────────


def _state_dir(config: ProjectConfig) -> Path:
    return config.root / STATE_DIR


def _last_run_path(config: ProjectConfig) -> Path:
    return _state_dir(config) / LAST_RUN


def _ensure_state_dir(config: ProjectConfig) -> None:
    d = _state_dir(config)
    d.mkdir(parents=True, exist_ok=True)
    gitignore = d / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("# created by fasteval\n*\n", encoding="utf-8")


def _load_last_run(config: ProjectConfig) -> Optional[Dict[str, Any]]:
    path = _last_run_path(config)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_meta(config: ProjectConfig) -> Dict[str, Any]:
    path = _state_dir(config) / LAST_RUN_META
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


# ── commands ─────────────────────────────────────────────────────────────────


def _resolve_profile(config: ProjectConfig, name: Optional[str]) -> Profile:
    return config.resolve(name)


def _profile_label(config: ProjectConfig, profile: Profile) -> str:
    return f"{ui.style(profile.name, 'bold', 'cyan')} {ui.style('(' + config.source_label + ')', 'dim')}"


def _print_plan(config: ProjectConfig, plan: Plan) -> None:
    dim = lambda s: ui.style(s, "dim")  # noqa: E731
    rows = [("profile", _profile_label(config, plan.profile))]
    rows.append(
        ("paths", " ".join(plan.paths) if plan.paths else dim("pytest testpaths"))
    )
    rows.append(("select", plan.marker_expr or dim("all evals")))
    if plan.keyword_expr:
        rows.append(("name", plan.keyword_expr))
    extras = []
    if plan.workers:
        extras.append(f"{plan.workers} workers")
    if not plan.review_prompts:
        extras.append("no review prompts")
    rows.append(("mode", " · ".join(extras)))
    gates = []
    if plan.min_score is not None:
        gates.append(f"score ≥ {plan.min_score:.2f}")
    if plan.min_pass_rate is not None:
        gates.append(f"pass rate ≥ {plan.min_pass_rate:.0%}")
    if gates:
        rows.append(("gates", " · ".join(gates)))
    if plan.judges:
        rows.append(("judges", " ".join(plan.judges)))
    if plan.outputs:
        rows.append(("reports", " ".join(o.split(":", 1)[1] for o in plan.outputs)))
    env_keys = [k for k in plan.env if k != "FASTEVAL_PROFILE"]
    if env_keys:
        rows.append(("env", " ".join(env_keys)))

    print(ui.style("fasteval", "bold", "magenta") + dim(" · run"))
    for key, value in rows:
        print(f"  {dim(f'{key:<8}')} {value}")
    print(f"  {dim('$ ' + plan.display)}\n")


def resolve_headless(args: argparse.Namespace, profile: Profile) -> bool:
    """UI mode unless told otherwise or there's nobody to look at it."""
    if args.ui:
        return False
    if args.headless or args.ci or is_ci_env():
        return True
    if profile.headless is not None:
        return profile.headless
    return not (sys.stdout.isatty() and sys.stdin.isatty())


def cmd_run(config: ProjectConfig, args: argparse.Namespace, extra: List[str]) -> int:
    from fasteval.ui import runs as runs_mod

    profile = _resolve_profile(config, args.profile)
    plan = build_run_plan(config, profile, args, extra)

    if plan.workers and importlib.util.find_spec("xdist") is None:
        ui.error(
            f"workers={plan.workers} needs pytest-xdist. "
            "Install it with: pip install pytest-xdist"
        )
        return 2

    if args.dry_run:
        print(shlex.join(plan.argv))
        return 0

    headless = resolve_headless(args, profile)
    app = None
    if not headless:
        from fasteval.ui.export import bundle_available

        if bundle_available():
            from fasteval.ui.server import UIApp

            app = UIApp(config, port=args.port)
            app.bind()
            app.serve_in_thread()
        else:
            ui.error(
                "UI bundle not built; running headless (cd ui && npm ci && npm run build)"
            )
            headless = True

    _print_plan(config, plan)
    _ensure_state_dir(config)
    request = runs_mod.RunRequest.from_args(args, extra)
    run_id, run_path, proc, meta = runs_mod.launch(
        config,
        request,
        origin="cli",
        capture_output=False,
        force_non_interactive=bool(args.ci),
    )
    if app is not None:
        url = f"{app.url}#/runs/{run_id}"
        print(ui.style("  ui      ", "dim") + ui.style(url, "cyan") + "\n")
        if not args.no_open:
            app.open_browser_when_ready(url)
    sys.stdout.flush()

    try:
        code = proc.wait()
    except KeyboardInterrupt:
        code = proc.wait()

    (_state_dir(config) / LAST_RUN_META).write_text(
        json.dumps({**meta, "exit_code": code}, indent=2), encoding="utf-8"
    )

    if headless:
        _write_static_report(config, run_path, args.html)
        return code
    if args.html:  # an explicit report path is honored in UI mode too
        _write_static_report(config, run_path, args.html)

    assert app is not None
    print(
        "\n"
        + ui.style("fasteval", "bold", "magenta")
        + f" · run finished · UI at {ui.style(app.url + '#/runs/' + run_id, 'cyan')}"
    )
    print(ui.style("  re-run, compare and tune judges there · Ctrl+C to exit", "dim"))
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
    return code


def _write_static_report(
    config: ProjectConfig, run_path: Path, target: Optional[str]
) -> Optional[Path]:
    from fasteval.ui.export import UIBundleMissing, write_report

    out = Path(target) if target else config.root / STATE_DIR / "report.html"
    try:
        write_report(config, run_path, out)
    except UIBundleMissing as e:
        ui.error(str(e))
        return None
    print(ui.style("  report  ", "dim") + _display_path(str(out)))
    return out


def cmd_ui(config: ProjectConfig, args: argparse.Namespace, extra: List[str]) -> int:
    from fasteval.ui.export import BUILD_HINT, bundle_available
    from fasteval.ui.server import UIApp

    if not bundle_available():
        ui.error(BUILD_HINT)
        return 2
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        ui.warn(
            f"binding to {args.host}: anyone who can reach this address can view runs "
            "and judge prompts. The session token only protects actions."
        )
    app = UIApp(config, host=args.host, port=args.port)
    url = app.bind()
    print(ui.style("fasteval", "bold", "magenta") + f" · UI at {ui.style(url, 'cyan')}")
    print(ui.style(f"  project {config.root} · Ctrl+C to exit", "dim"))
    if not args.no_open:
        app.open_browser_when_ready(url)
    try:
        assert app.httpd is not None
        app.httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
    return 0


def collect_inventory(
    config: ProjectConfig, profile: Profile, args: argparse.Namespace
) -> tuple[Optional[Dict[str, Any]], str]:
    """Run pytest --collect-only and return (inventory, pytest output)."""
    selection, *_ = build_selection(profile, args)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "inventory.json"
        argv = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--fe-inventory={out}",
            *selection,
            *profile.pytest_args,
        ]
        env = {**os.environ, **profile.env, "FASTEVAL_PROFILE": profile.name}
        proc = subprocess.run(argv, env=env, capture_output=True, text=True)
        output = proc.stdout + proc.stderr
        # 5 = no tests collected; still a valid (empty) inventory.
        if proc.returncode not in (0, 5) or not out.exists():
            return None, output
        return json.loads(out.read_text(encoding="utf-8")), output


def _metric_label(m: Dict[str, Any]) -> str:
    return f"{m['name']}≥{m['threshold']:g}"


def _data_label(data: Optional[Dict[str, Any]]) -> str:
    if not data:
        return ""
    kind = data.get("type", "data")
    if kind in ("csv", "jsonl"):
        return f"{kind} {data.get('path', '')}"
    if kind == "dataset":
        return f"dataset {data.get('name', '')}"
    if kind == "conversation":
        return f"conversation {data.get('turns', '?')} turns"
    if kind == "langfuse_dataset":
        return f"langfuse dataset {data.get('name', '')}"
    if kind == "langfuse_traces":
        return f"langfuse traces {data.get('project', '')}"
    return kind


def cmd_list(config: ProjectConfig, args: argparse.Namespace, extra: List[str]) -> int:
    profile = _resolve_profile(config, args.profile)
    inventory, output = collect_inventory(config, profile, args)
    if inventory is None:
        ui.error("collection failed:\n" + output.strip())
        return 1

    items = inventory["items"]
    if args.as_json:
        print(json.dumps(items, indent=2))
        return 0

    # Group parametrized / data-driven cases under their function.
    files: "OrderedDict[str, OrderedDict[str, List[Dict[str, Any]]]]" = OrderedDict()
    for item in items:
        funcs = files.setdefault(item["file"], OrderedDict())
        funcs.setdefault(item["function"], []).append(item)

    # Count tags per eval (function), matching the rows shown below.
    tag_counts: Counter = Counter(
        t
        for funcs in files.values()
        for cases in funcs.values()
        for t in cases[0]["tags"]
    )
    if args.tags_only:
        return _print_tags(config, tag_counts)

    dim = lambda s: ui.style(s, "dim")  # noqa: E731
    print(
        ui.style("fasteval", "bold", "magenta")
        + dim(" · list · ")
        + _profile_label(config, profile)
    )
    if not items:
        print("\n  No evals found. Evals are tests that use fasteval decorators,")
        print("  e.g. @fe.correctness() or @fe.contains().")
        return 0

    # Build plain-text cells first so widths ignore ANSI codes.
    rows: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    for file, funcs in files.items():
        for func, cases in funcs.items():
            first = cases[0]
            extras = [_data_label(first["data"])]
            if first["human_review"]:
                extras.append("human review")
            if first["skipped"]:
                extras.append("skipped")
            rows.setdefault(file, []).append(
                {
                    "name": func,
                    "count": f" ×{len(cases)}" if len(cases) > 1 else "",
                    "tags": " ".join(first["tags"]),
                    "metrics": " ".join(_metric_label(m) for m in first["metrics"]),
                    "extra": " · ".join(e for e in extras if e),
                }
            )
    all_rows = [r for rs in rows.values() for r in rs]
    name_w = max(len(r["name"] + r["count"]) for r in all_rows)
    tag_w = max(len(r["tags"]) for r in all_rows)

    for file, file_rows in rows.items():
        print("\n" + ui.style(file, "bold"))
        for r in file_rows:
            pad = " " * (name_w - len(r["name"] + r["count"]))
            extra_cell = dim("  " + r["extra"]) if r["extra"] else ""
            print(
                f"  {r['name']}{dim(r['count'])}{pad}  "
                f"{ui.style(r['tags'].ljust(tag_w), 'cyan')}  {r['metrics']}{extra_cell}"
            )

    summary = f"{len(all_rows)} evals"
    if len(items) != len(all_rows):
        summary += f" ({len(items)} cases)"
    summary += f" in {len(files)} file" + ("s" if len(files) != 1 else "")
    if tag_counts:
        summary += " · " + " ".join(f"{t}({n})" for t, n in tag_counts.most_common())
    print("\n" + dim(summary))
    return 0


def _print_tags(config: ProjectConfig, counts: Counter) -> int:
    names = sorted(set(counts) | set(config.tags))
    if not names:
        print("No tags. Add pytest markers to evals, e.g. @pytest.mark.smoke")
        return 0
    w = max(len(n) for n in names)
    for name in names:
        desc = config.tags.get(name, ui.style("(not registered)", "yellow"))
        print(f"  {ui.style(name.ljust(w), 'cyan')}  {counts.get(name, 0):>4}  {desc}")
    return 0


def cmd_profiles(
    config: ProjectConfig, args: argparse.Namespace, extra: List[str]
) -> int:
    if args.name:
        profile = config.resolve(args.name)
        ns = build_parser().parse_args(["run"])
        plan = build_run_plan(config, profile, ns)
        _print_plan(config, plan)
        return 0

    default = config.default_profile_name()
    names = config.profile_names()
    w = max(len(n) for n in names)
    print(
        ui.style("fasteval", "bold", "magenta")
        + ui.style(f" · profiles ({config.source_label})", "dim")
    )
    for name in names:
        raw = config.profiles.get(name, {})
        marker = ui.style("*", "green") if name == default else " "
        desc = raw.get("description", "")
        if not desc and name == DEFAULT_PROFILE and name not in config.profiles:
            desc = ui.style("built-in: all evals under pytest testpaths", "dim")
        if raw.get("extends"):
            desc = (desc + " " if desc else "") + ui.style(
                f"(extends {raw['extends']})", "dim"
            )
        print(f" {marker} {ui.style(name.ljust(w), 'cyan')}  {desc}")
    print(ui.style("\n  fasteval profiles NAME   show what a profile runs", "dim"))
    return 0


def _print_summary(summary: Dict[str, Any], meta: Dict[str, Any]) -> None:
    dim = lambda s: ui.style(s, "dim")  # noqa: E731
    total = summary.get("total_tests", 0)
    passed = summary.get("passed_tests", 0)
    failed = summary.get("failed_tests", 0)
    rate = summary.get("pass_rate", 0.0)
    score = summary.get("avg_aggregate_score", 0.0)

    header = ui.style("fasteval", "bold", "magenta") + dim(" · last run")
    if meta.get("profile"):
        header += dim(" · profile ") + ui.style(meta["profile"], "cyan")
    if meta.get("started_at"):
        header += dim(" · " + _ago(meta["started_at"]))
    print(header)
    print(
        f"  {ui.style(str(passed) + ' passed', 'green')}  "
        f"{ui.style(str(failed) + ' failed', 'red' if failed else 'dim')}  "
        f"{dim(f'of {total}')}   pass rate {ui.style(ui.pct(rate), ui.score_style(rate, 0.9))}"
        f"   avg score {ui.style(f'{score:.3f}', ui.score_style(score))}"
    )
    aggs = summary.get("metric_aggregates", [])
    if aggs:
        w = max(len(m["metric_name"]) for m in aggs)
        print()
        for m in aggs:
            bar = _bar(m["avg_score"])
            print(
                f"  {m['metric_name'].ljust(w)}  {bar} {m['avg_score']:.3f}  "
                f"{dim('pass')} {ui.pct(m['pass_rate']):>4}  "
                + dim(f"min {m['min_score']:.2f} max {m['max_score']:.2f}")
            )
    fails = [t for t in summary.get("test_summaries", []) if not t["passed"]]
    if fails:
        print()
        for t in fails:
            err = dim(f"  {t['error']}") if t.get("error") else ""
            print(
                f"  {ui.style('✗', 'red')} {t['test_name']}  {t['aggregate_score']:.3f}{err}"
            )
    if meta.get("command"):
        print(dim(f"\n  $ {meta['command']}"))


def _bar(value: float, width: int = 16) -> str:
    filled = max(0, min(width, round(value * width)))
    return ui.style("█" * filled, ui.score_style(value)) + ui.style(
        "░" * (width - filled), "dim"
    )


def _ago(iso: str) -> str:
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    secs = int((datetime.now(timezone.utc) - then).total_seconds())
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= size:
            return f"{secs // size}{unit} ago"
    return "just now"


def cmd_report(
    config: ProjectConfig, args: argparse.Namespace, extra: List[str]
) -> int:
    from fasteval.ui import runs as runs_mod

    runs = runs_mod.list_runs(config)
    if args.run_id:
        run_path = runs_mod.run_dir(config, args.run_id)
        if run_path is None:
            ui.error(f"Run {args.run_id!r} not found. See: fasteval report --list")
            return 1
    elif runs:
        run_path = runs_mod.runs_root(config) / runs[0]["id"]
    else:
        print("No runs yet. Start one with: fasteval run")
        return 1

    if args.list:
        for r in runs[:20]:
            rate = f"{r['pass_rate']:.0%}" if r.get("pass_rate") is not None else "  — "
            print(f"  {r['id']}  {r['status']:<11} {rate:>4}  {r.get('profile') or ''}")
        return 0

    result = runs_mod.read_json(run_path / "result.json")
    meta = runs_mod.read_json(run_path / runs_mod.META_FILE)
    if result.get("summary"):
        _print_summary(result["summary"], meta)
    else:
        print(
            f"Run {run_path.name} has no results ({runs_mod.summarize_run(run_path)['status']})."
        )

    if args.html is not None or args.open:
        out = _write_static_report(config, run_path, args.html or None)
        if out is not None and args.open:
            webbrowser.open(out.resolve().as_uri())
    return 0


def cmd_status(config: ProjectConfig) -> int:
    dim = lambda s: ui.style(s, "dim")  # noqa: E731
    from fasteval import __version__

    print(ui.style("fasteval", "bold", "magenta") + dim(f" {__version__}"))
    print(f"  {dim('project ')} {_display_path(str(config.root))}")
    print(f"  {dim('config  ')} {config.source_label}")
    names = config.profile_names()
    default = config.default_profile_name()
    print(
        f"  {dim('profiles')} "
        + " ".join(ui.style(n + ("*" if n == default else ""), "cyan") for n in names)
    )
    data = _load_last_run(config)
    if data:
        s = data.get("summary", {})
        meta = _load_meta(config)
        when = _ago(meta["started_at"]) if meta.get("started_at") else ""
        print(
            f"  {dim('last run')} {s.get('passed_tests', 0)}/{s.get('total_tests', 0)} passed"
            f" · avg {s.get('avg_aggregate_score', 0):.3f}"
            + (f" · {meta.get('profile')}" if meta.get("profile") else "")
            + (f" · {when}" if when else "")
        )
    print(
        dim(
            "\n  fasteval run       run evals, live UI   fasteval list      browse evals"
        )
    )
    print(
        dim(
            "  fasteval ui        runs, judges, play   fasteval report    summary / HTML"
        )
    )
    print(
        dim("  fasteval profiles  see profiles         fasteval run --headless  for CI")
    )
    if config.source is None:
        print(dim("  fasteval init     add a [tool.fasteval] config"))
    return 0


_STARTER = """
[tool.fasteval]
default_profile = "dev"

# Tags are pytest markers: @pytest.mark.smoke. Registering them documents
# them in `fasteval list --tags` and silences unknown-marker warnings.
[tool.fasteval.tags]
smoke = "Fast sanity evals"
slow = "Long-running or expensive evals"

[tool.fasteval.profiles.dev]
description = "Local iteration: everything but slow evals"
paths = ["{paths}"]
tags = "not slow"

[tool.fasteval.profiles.ci]
description = "CI gate: all evals, headless, JSON + HTML report"
paths = ["{paths}"]
review_prompts = false
headless = true
outputs = ["json:.fasteval/results.json"]
min_pass_rate = 0.9
"""


def cmd_init(config: ProjectConfig) -> int:
    if config.source is not None:
        print(f"Already configured in {config.source_label}.")
        return 0
    root = config.root
    paths = next((d for d in ("evals", "tests") if (root / d).is_dir()), "tests")
    block = _STARTER.format(paths=paths)
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        sep = "" if text.endswith("\n") else "\n"
        pyproject.write_text(text + sep + block, encoding="utf-8")
        target = pyproject
    else:
        target = root / "fasteval.toml"
        body = block.replace("[tool.fasteval.", "[").replace("[tool.fasteval]\n", "")
        target.write_text(body.lstrip(), encoding="utf-8")
    print(f"Added fasteval config to {_display_path(str(target))}")
    print(ui.style("  next: fasteval profiles · fasteval list · fasteval run", "dim"))
    return 0


# ── entry point ──────────────────────────────────────────────────────────────

_PASSTHROUGH_COMMANDS = {"run"}


def _reclaim_option_values(
    argv: List[str], args: argparse.Namespace, unknown: List[str]
) -> List[str]:
    """
    Give values of pass-through pytest options back to their option.

    argparse can't know that ``-W error`` or ``--tb short`` take a value, so
    it files ``error``/``short`` under positional paths. A "path" that
    doesn't exist on disk and directly follows an unknown option is
    re-attached to that option.
    """
    unknown_set = set(unknown)
    out = list(unknown)
    for i, token in enumerate(argv[1:], start=1):
        prev = argv[i - 1]
        if (
            token in args.paths
            and prev in unknown_set
            and prev.startswith("-")
            and "=" not in prev
            and not Path(token.split("::", 1)[0]).exists()
        ):
            args.paths.remove(token)
            out.insert(out.index(prev) + 1, token)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)

    # Everything after a bare `--` always goes to pytest untouched.
    explicit: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, explicit = argv[:i], argv[i + 1 :]

    args, unknown = parser.parse_known_args(argv)
    if args.command == "run" and unknown:
        unknown = _reclaim_option_values(argv, args, unknown)

    if args.version:
        from fasteval import __version__

        print(f"fasteval {__version__}")
        return 0

    if unknown and args.command not in _PASSTHROUGH_COMMANDS:
        parser.error(f"unrecognized arguments: {' '.join(unknown)}")

    try:
        config = find_config()
        if args.command is None:
            return cmd_status(config)
        if args.command == "run":
            return cmd_run(config, args, unknown + explicit)
        if args.command in ("list", "ls"):
            return cmd_list(config, args, [])
        if args.command == "profiles":
            return cmd_profiles(config, args, [])
        if args.command == "report":
            return cmd_report(config, args, [])
        if args.command == "init":
            return cmd_init(config)
        if args.command == "ui":
            return cmd_ui(config, args, [])
    except ConfigError as e:
        ui.error(str(e))
        return 2
    parser.print_help()  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
