"""
Project configuration and profiles for the fasteval CLI.

Configuration lives in ``pyproject.toml`` under ``[tool.fasteval]`` or in a
standalone ``fasteval.toml`` (same keys, no ``tool.fasteval`` prefix)::

    [tool.fasteval]
    default_profile = "dev"

    # Tag registry. Tags are plain pytest markers; registering them here
    # documents them and silences PytestUnknownMarkWarning.
    [tool.fasteval.tags]
    smoke = "Fast sanity evals, run on every change"
    rag = "Retrieval quality"
    slow = "Long-running or expensive evals"

    [tool.fasteval.profiles.dev]
    paths = ["evals"]
    tags = "not slow"

    [tool.fasteval.profiles.ci]
    extends = "dev"
    tags = ""                       # clear the inherited selection
    review_prompts = false          # never stop for human-review input
    headless = true                 # no UI server; write a static report
    workers = "auto"
    outputs = ["json:.fasteval/results.json"]   # headless runs also write .fasteval/report.html
    min_pass_rate = 0.9
    min_score = 0.75
    env = { OPENAI_MODEL = "gpt-4o-mini" }

Inheritance via ``extends``: scalar and list values from the child replace
the parent's; ``env`` tables are merged.
"""

from __future__ import annotations

import difflib
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

PYPROJECT = "pyproject.toml"
FASTEVAL_TOML = "fasteval.toml"
DEFAULT_PROFILE = "default"

_TOP_LEVEL_KEYS = {"default_profile", "tags", "profiles", "judges_dir", "judge_client"}


class ConfigError(Exception):
    """Raised for invalid fasteval configuration."""


@dataclass
class Profile:
    """A named, fully-resolved set of run settings."""

    name: str
    description: str = ""
    paths: List[str] = field(default_factory=list)
    tags: Optional[str] = None
    keyword: Optional[str] = None
    workers: Optional[Union[int, str]] = None
    outputs: List[str] = field(default_factory=list)
    review_prompts: Optional[bool] = None
    headless: Optional[bool] = None
    min_score: Optional[float] = None
    min_pass_rate: Optional[float] = None
    env: Dict[str, str] = field(default_factory=dict)
    pytest_args: List[str] = field(default_factory=list)


# Keys a profile table may contain (``extends`` is resolved, not stored).
PROFILE_KEYS = {f.name for f in fields(Profile)} - {"name"} | {"extends"}


@dataclass
class ProjectConfig:
    """Parsed ``[tool.fasteval]`` configuration for a project."""

    root: Path
    source: Optional[Path] = None
    default_profile: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    judges_dir: Optional[str] = None
    judge_client: Optional[str] = None

    @property
    def source_label(self) -> str:
        return self.source.name if self.source else "built-in defaults"

    def profile_names(self) -> List[str]:
        names = list(self.profiles)
        if DEFAULT_PROFILE not in names:
            names.insert(0, DEFAULT_PROFILE)
        return names

    def default_profile_name(self) -> str:
        return self.default_profile or DEFAULT_PROFILE

    def resolve(self, name: Optional[str] = None) -> Profile:
        """Resolve a profile by name, applying ``extends`` inheritance."""
        name = name or self.default_profile_name()
        merged = self._merge_chain(name, [])
        merged.pop("extends", None)
        profile = Profile(name=name, **merged)
        profile.paths = [self._resolve_path(p) for p in profile.paths]
        return profile

    def _merge_chain(self, name: str, seen: List[str]) -> Dict[str, Any]:
        if name in seen:
            chain = " -> ".join(seen + [name])
            raise ConfigError(f"Profile inheritance cycle: {chain}")
        if name not in self.profiles:
            if name == DEFAULT_PROFILE:
                return {}
            raise ConfigError(
                f"Unknown profile {name!r}."
                + _did_you_mean(name, self.profile_names())
                + f" Available: {', '.join(self.profile_names())}"
            )
        raw = dict(self.profiles[name])
        parent_name = raw.get("extends")
        if not parent_name:
            return raw
        parent = self._merge_chain(parent_name, seen + [name])
        env = {**parent.get("env", {}), **raw.get("env", {})}
        merged = {**parent, **raw}
        if env:
            merged["env"] = env
        return merged

    def _resolve_path(self, path: str) -> str:
        p = Path(path)
        if p.is_absolute():
            return path
        # Keep pytest node ids (path::test) intact while anchoring at root.
        return str(self.root / path)


def find_config(start: Optional[Path] = None) -> ProjectConfig:
    """
    Locate and load configuration, walking up from ``start`` (default: cwd).

    The first directory containing ``fasteval.toml`` or ``pyproject.toml``
    is the project root. If none is found, ``start`` is the root and
    built-in defaults apply.
    """
    start = (start or Path.cwd()).resolve()
    for directory in [start, *start.parents]:
        fe_toml = directory / FASTEVAL_TOML
        if fe_toml.is_file():
            return _from_table(directory, fe_toml, _read_toml(fe_toml))
        pyproject = directory / PYPROJECT
        if pyproject.is_file():
            table = _read_toml(pyproject).get("tool", {}).get("fasteval")
            if table is None:
                return ProjectConfig(root=directory)
            return _from_table(directory, pyproject, table)
    return ProjectConfig(root=start)


def load_tag_registry(root: Path) -> Dict[str, str]:
    """
    Read only the tag registry for ``root``. Used by the pytest plugin, so it
    never raises for problems unrelated to tags.
    """
    try:
        config = find_config(root)
    except ConfigError:
        return {}
    return config.tags


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e


def _from_table(root: Path, source: Path, table: Dict[str, Any]) -> ProjectConfig:
    where = (
        f"{source.name} [tool.fasteval]" if source.name == PYPROJECT else source.name
    )
    _check_keys(table, _TOP_LEVEL_KEYS, where)

    tags = table.get("tags", {})
    if not isinstance(tags, dict) or not all(isinstance(v, str) for v in tags.values()):
        raise ConfigError(f"{where}: 'tags' must be a table of name = \"description\"")

    profiles = table.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ConfigError(f"{where}: 'profiles' must be a table")
    for name, body in profiles.items():
        if not isinstance(body, dict):
            raise ConfigError(f"{where}: profile {name!r} must be a table")
        _validate_profile(name, body, f"{where} profiles.{name}")

    default = table.get("default_profile")
    if default is not None and default != DEFAULT_PROFILE and default not in profiles:
        raise ConfigError(
            f"{where}: default_profile {default!r} is not defined."
            + _did_you_mean(default, list(profiles))
        )

    for key in ("judges_dir", "judge_client"):
        if key in table and not isinstance(table[key], str):
            raise ConfigError(f"{where}: '{key}' must be a string")
    judge_client = table.get("judge_client")
    if judge_client is not None and ":" not in judge_client:
        raise ConfigError(
            f"{where}: 'judge_client' must look like 'package.module:attribute'"
        )

    return ProjectConfig(
        root=root,
        source=source,
        default_profile=default,
        tags=dict(tags),
        profiles={k: dict(v) for k, v in profiles.items()},
        judges_dir=table.get("judges_dir"),
        judge_client=judge_client,
    )


def _validate_profile(name: str, body: Dict[str, Any], where: str) -> None:
    _check_keys(body, PROFILE_KEYS, where)

    def expect(key: str, types: tuple, label: str) -> None:
        if key in body and not isinstance(body[key], types):
            raise ConfigError(f"{where}: '{key}' must be {label}")

    def expect_str_list(key: str) -> None:
        value = body.get(key)
        if value is not None and (
            not isinstance(value, list) or not all(isinstance(v, str) for v in value)
        ):
            raise ConfigError(f"{where}: '{key}' must be a list of strings")

    for key in ("description", "tags", "keyword", "extends"):
        expect(key, (str,), "a string")
    expect("review_prompts", (bool,), "true or false")
    expect("headless", (bool,), "true or false")
    for key in ("paths", "outputs", "pytest_args"):
        expect_str_list(key)

    workers = body.get("workers")
    if workers is not None and not (
        (isinstance(workers, int) and not isinstance(workers, bool) and workers >= 0)
        or workers in ("auto", "logical")
    ):
        raise ConfigError(f"{where}: 'workers' must be a non-negative int or \"auto\"")

    for key in ("min_score", "min_pass_rate"):
        value = body.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
        ):
            raise ConfigError(f"{where}: '{key}' must be a number between 0 and 1")

    env = body.get("env")
    if env is not None and (
        not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values())
    ):
        raise ConfigError(f"{where}: 'env' must be a table of string values")

    for spec in body.get("outputs", []):
        fmt, _, path = spec.partition(":")
        if not fmt or not path:
            raise ConfigError(
                f"{where}: output {spec!r} must look like 'json:path' or 'html:path'"
            )


def _check_keys(table: Dict[str, Any], allowed: set, where: str) -> None:
    for key in table:
        if key not in allowed:
            raise ConfigError(
                f"{where}: unknown key {key!r}." + _did_you_mean(key, sorted(allowed))
            )


def _did_you_mean(word: str, options: List[str]) -> str:
    match = difflib.get_close_matches(word, options, n=1)
    return f" Did you mean {match[0]!r}?" if match else ""
