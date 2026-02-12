#!/usr/bin/env python3
"""Calculate the next version for a package based on git tags and bump type.

Usage:
    python scripts/bump_version.py --package core --bump alpha
    python scripts/bump_version.py --package langfuse --bump minor
    python scripts/bump_version.py --package core --bump dev --pr-number 42

Outputs GitHub Actions step outputs to stdout (append to $GITHUB_OUTPUT):
    version=1.0.0a2
    tag=v1.0.0a2
    pyproject_path=pyproject.toml
    build_dir=.
    current_version=1.0.0a1
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Package registry
# ---------------------------------------------------------------------------

PACKAGES: dict[str, dict[str, str]] = {
    "core": {
        "tag_prefix": "v",
        "pyproject_path": "pyproject.toml",
        "build_dir": ".",
    },
    "langfuse": {
        "tag_prefix": "langfuse-v",
        "pyproject_path": "plugins/fasteval-langfuse/pyproject.toml",
        "build_dir": "plugins/fasteval-langfuse",
    },
    "observe": {
        "tag_prefix": "observe-v",
        "pyproject_path": "plugins/fasteval-observe/pyproject.toml",
        "build_dir": "plugins/fasteval-observe",
    },
    "langgraph": {
        "tag_prefix": "langgraph-v",
        "pyproject_path": "plugins/fasteval-langgraph/pyproject.toml",
        "build_dir": "plugins/fasteval-langgraph",
    },
}

BUMP_TYPES = ("major", "minor", "patch", "alpha", "beta", "rc", "dev", "stable")

# ---------------------------------------------------------------------------
# Version parsing / formatting
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?(?:\.dev(\d+))?$")


@dataclass
class Version:
    major: int
    minor: int
    patch: int
    pre_type: str | None = None  # "a", "b", "rc", or None
    pre_num: int | None = None
    dev_num: int | None = None

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        m = _VERSION_RE.match(version_str)
        if not m:
            raise ValueError(f"Cannot parse PEP 440 version: {version_str!r}")
        return cls(
            major=int(m.group(1)),
            minor=int(m.group(2)),
            patch=int(m.group(3)),
            pre_type=m.group(4),
            pre_num=int(m.group(5)) if m.group(5) else None,
            dev_num=int(m.group(6)) if m.group(6) else None,
        )

    @property
    def is_prerelease(self) -> bool:
        return self.pre_type is not None or self.dev_num is not None

    def __str__(self) -> str:
        v = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_type is not None and self.pre_num is not None:
            v += f"{self.pre_type}{self.pre_num}"
        if self.dev_num is not None:
            v += f".dev{self.dev_num}"
        return v


# ---------------------------------------------------------------------------
# Bump logic
# ---------------------------------------------------------------------------


def bump(current: Version, bump_type: str, *, pr_number: int | None = None) -> Version:
    """Return the next version after applying *bump_type* to *current*."""
    maj, min_, pat = current.major, current.minor, current.patch

    if bump_type == "major":
        return Version(major=maj + 1, minor=0, patch=0)

    if bump_type == "minor":
        return Version(major=maj, minor=min_ + 1, patch=0)

    if bump_type == "patch":
        if current.is_prerelease:
            # From pre-release, patch promotes to stable (same as "stable")
            return Version(major=maj, minor=min_, patch=pat)
        return Version(major=maj, minor=min_, patch=pat + 1)

    if bump_type == "alpha":
        if current.pre_type == "a":
            return Version(maj, min_, pat, "a", (current.pre_num or 0) + 1)
        if current.pre_type in ("b", "rc"):
            raise ValueError(
                f"Cannot create alpha from {current.pre_type} release ({current})"
            )
        # From stable: start next minor alpha
        return Version(maj, min_ + 1, 0, "a", 1)

    if bump_type == "beta":
        if current.pre_type == "b":
            return Version(maj, min_, pat, "b", (current.pre_num or 0) + 1)
        if current.pre_type == "a":
            return Version(maj, min_, pat, "b", 1)
        if current.pre_type == "rc":
            raise ValueError(f"Cannot create beta from rc release ({current})")
        # From stable: start next minor beta
        return Version(maj, min_ + 1, 0, "b", 1)

    if bump_type == "rc":
        if current.pre_type == "rc":
            return Version(maj, min_, pat, "rc", (current.pre_num or 0) + 1)
        if current.pre_type in ("a", "b"):
            return Version(maj, min_, pat, "rc", 1)
        # From stable: start next minor rc
        return Version(maj, min_ + 1, 0, "rc", 1)

    if bump_type == "dev":
        if pr_number is None:
            raise ValueError("--pr-number is required for dev bumps")
        # Dev version: next minor with .devN where N = PR number
        if current.is_prerelease:
            return Version(maj, min_ + 1, 0, dev_num=pr_number)
        return Version(maj, min_ + 1, 0, dev_num=pr_number)

    if bump_type == "stable":
        if not current.is_prerelease:
            raise ValueError(f"Already a stable version ({current})")
        return Version(major=maj, minor=min_, patch=pat)

    raise ValueError(f"Unknown bump type: {bump_type!r}")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def get_latest_tag(prefix: str) -> str | None:
    """Return the most recent git tag matching *prefix*, or None."""
    result = subprocess.run(
        ["git", "tag", "--list", f"{prefix}*", "--sort=-v:refname"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.strip().splitlines():
        tag = line.strip()
        if tag:
            return tag
    return None


def read_pyproject_version(path: str) -> str:
    """Read the version string from a pyproject.toml file."""
    version_re = re.compile(r'^version\s*=\s*"(.+)"')
    with open(path) as f:
        for line in f:
            m = version_re.match(line)
            if m:
                return m.group(1)
    raise ValueError(f"Could not find version in {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the next PEP 440 version for a package."
    )
    parser.add_argument(
        "--package",
        required=True,
        choices=sorted(PACKAGES.keys()),
        help="Package to bump.",
    )
    parser.add_argument(
        "--bump",
        required=True,
        choices=BUMP_TYPES,
        help="Bump type to apply.",
    )
    parser.add_argument(
        "--pr-number",
        type=int,
        default=None,
        help="PR number (required for dev bumps).",
    )
    args = parser.parse_args()

    pkg = PACKAGES[args.package]
    prefix = pkg["tag_prefix"]

    # Determine current version from latest tag, fallback to pyproject.toml
    tag = get_latest_tag(prefix)
    if tag:
        current_str = tag[len(prefix) :]
        source = f"tag {tag}"
    else:
        current_str = read_pyproject_version(pkg["pyproject_path"])
        source = pkg["pyproject_path"]

    current = Version.parse(current_str)
    next_ver = bump(current, args.bump, pr_number=args.pr_number)
    tag_name = f"{prefix}{next_ver}"

    # Log to stderr (visible in CI logs but not in $GITHUB_OUTPUT)
    print(f"[bump_version] package={args.package}", file=sys.stderr)
    print(f"[bump_version] source={source}", file=sys.stderr)
    print(f"[bump_version] {current} --{args.bump}--> {next_ver}", file=sys.stderr)

    # Output for GitHub Actions (append to $GITHUB_OUTPUT)
    print(f"version={next_ver}")
    print(f"tag={tag_name}")
    print(f"pyproject_path={pkg['pyproject_path']}")
    print(f"build_dir={pkg['build_dir']}")
    print(f"current_version={current}")


if __name__ == "__main__":
    main()
