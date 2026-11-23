"""
Self-contained HTML report of a run (``fasteval run --headless``, CI).

The report is the same React UI as ``fasteval ui``, read-only, with the CSS,
JS and the run's full event log inlined into one file. It works offline
from a CI artifact viewer or ``file://``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fasteval.cli.config import ProjectConfig
from fasteval.ui import runs as runs_mod

DIST_DIR = Path(__file__).parent / "static" / "dist"

BUILD_HINT = (
    "The fasteval UI bundle isn't built. From a source checkout run:\n"
    "    cd ui && npm ci && npm run build\n"
    "(Released fasteval-core wheels include it.)"
)


class UIBundleMissing(RuntimeError):
    pass


def bundle_available() -> bool:
    return (DIST_DIR / "index.html").is_file() and (DIST_DIR / "app.js").is_file()


def project_info(config: ProjectConfig) -> Dict[str, Any]:
    """Project summary used by the UI (profiles, tags, judges)."""
    from fasteval.judges import list_judges

    profiles = []
    for name in config.profile_names():
        p = config.resolve(name)
        raw = config.profiles.get(name, {})
        profiles.append(
            {
                "name": name,
                "description": p.description,
                "extends": raw.get("extends"),
                "paths": [_rel(config, x) for x in p.paths],
                "tags": p.tags,
                "keyword": p.keyword,
                "workers": p.workers,
                "min_score": p.min_score,
                "min_pass_rate": p.min_pass_rate,
            }
        )
    return {
        "name": config.root.name,
        "root": str(config.root),
        "config_source": config.source_label,
        "default_profile": config.default_profile_name(),
        "profiles": profiles,
        "tags": config.tags,
        "judge_client": config.judge_client,
        "judges_dir": config.judges_dir or "judges",
        "judges": [j.name for j in list_judges(config.root)],
    }


def _rel(config: ProjectConfig, path: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.root))
    except ValueError:
        return path


def read_events(path: Path) -> List[Dict[str, Any]]:
    """All events of a run, merged across worker files, in time order."""
    events: List[Dict[str, Any]] = []
    for file in sorted(path.glob("events*.jsonl")):
        with file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    events.sort(key=lambda e: e.get("ts", 0))
    return events


def _script_json(data: Any) -> str:
    """JSON safe to embed in a <script> element."""
    return (
        json.dumps(data, default=str)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build_report(config: ProjectConfig, run_path: Path) -> str:
    """Render the static, single-file HTML report for one run."""
    if not bundle_available():
        raise UIBundleMissing(BUILD_HINT)
    run = runs_mod.summarize_run(run_path)
    run["meta"] = runs_mod.read_json(run_path / runs_mod.META_FILE)
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": project_info(config),
        "run": run,
        "events": read_events(run_path),
    }
    html = (DIST_DIR / "index.html").read_text(encoding="utf-8")
    css = (DIST_DIR / "app.css").read_text(encoding="utf-8")
    js = (
        (DIST_DIR / "app.js")
        .read_text(encoding="utf-8")
        .replace("</script", "<\\/script")
    )
    title = f"fasteval · {run['id']} · {run.get('status', '')}"

    html = html.replace("<title>fasteval</title>", f"<title>{_escape(title)}</title>")
    html = _drop_tag(html, '<link rel="stylesheet"')
    html = _drop_tag(html, '<script type="module"', closing="</script>")
    head_inline = f"<style>{css}</style>\n<script>window.FE_STATIC = {_script_json(data)};</script>"
    html = html.replace("</head>", head_inline + "\n</head>", 1)
    html = html.replace("</body>", f'<script type="module">{js}</script>\n</body>', 1)
    return html


def write_report(config: ProjectConfig, run_path: Path, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(config, run_path), encoding="utf-8")
    return out


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _drop_tag(html: str, start: str, closing: str = ">") -> str:
    i = html.find(start)
    if i < 0:
        return html
    j = html.find(closing, i)
    return html[:i] + html[j + len(closing) :] if j >= 0 else html
