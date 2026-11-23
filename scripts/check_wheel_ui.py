"""
Verify a fasteval-core wheel ships the built UI bundle, and only that.

Used by CI and the release workflow after `uv build`:

    python scripts/check_wheel_ui.py dist/

Fails (exit 1) when the bundle is missing (the UI would be broken for every
user of that release) or when UI source / node_modules leaked into the wheel.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

REQUIRED = (
    "fasteval/ui/static/dist/index.html",
    "fasteval/ui/static/dist/app.js",
    "fasteval/ui/static/dist/app.css",
)
FORBIDDEN_PARTS = ("node_modules/", ".tsx", ".ts", "ui/src/", "package.json")


def check(wheel: Path) -> list[str]:
    names = zipfile.ZipFile(wheel).namelist()
    problems = [f"missing {name}" for name in REQUIRED if name not in names]
    for name in names:
        if any(part in name for part in FORBIDDEN_PARTS) and not name.endswith(".py"):
            problems.append(f"unexpected UI source in wheel: {name}")
    return problems


def main(argv: list[str]) -> int:
    target = Path(argv[1] if len(argv) > 1 else "dist")
    wheels = sorted(target.glob("fasteval_core-*.whl")) if target.is_dir() else [target]
    if not wheels:
        print(f"No fasteval_core wheel found in {target}", file=sys.stderr)
        return 1
    failed = False
    for wheel in wheels:
        problems = check(wheel)
        if problems:
            failed = True
            print(f"{wheel.name}:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print(
                "  Build the UI first: cd ui && npm ci && npm run build",
                file=sys.stderr,
            )
        else:
            print(f"{wheel.name}: UI bundle present, no UI source")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
