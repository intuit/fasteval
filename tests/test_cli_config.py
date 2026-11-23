"""Tests for fasteval.cli.config (profiles and [tool.fasteval] loading)."""

from pathlib import Path

import pytest

from fasteval.cli.config import (
    ConfigError,
    find_config,
    load_tag_registry,
)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


PYPROJECT = """
[project]
name = "demo"

[tool.fasteval]
default_profile = "dev"

[tool.fasteval.tags]
smoke = "Fast checks"

[tool.fasteval.profiles.dev]
paths = ["evals"]
tags = "not slow"
env = { A = "1", B = "dev" }

[tool.fasteval.profiles.ci]
extends = "dev"
description = "CI"
review_prompts = false
workers = "auto"
min_pass_rate = 0.9
env = { B = "ci" }
outputs = ["json:out.json"]
"""


class TestFindConfig:
    def test_reads_pyproject(self, tmp_path):
        write(tmp_path / "pyproject.toml", PYPROJECT)
        config = find_config(tmp_path)
        assert config.root == tmp_path.resolve()
        assert config.source_label == "pyproject.toml"
        assert config.default_profile_name() == "dev"
        assert config.tags == {"smoke": "Fast checks"}

    def test_walks_up_from_subdirectory(self, tmp_path):
        write(tmp_path / "pyproject.toml", PYPROJECT)
        sub = tmp_path / "evals" / "nested"
        sub.mkdir(parents=True)
        assert find_config(sub).root == tmp_path.resolve()

    def test_fasteval_toml_wins_over_pyproject(self, tmp_path):
        write(tmp_path / "pyproject.toml", PYPROJECT)
        write(tmp_path / "fasteval.toml", '[profiles.only]\ntags = "x"\n')
        config = find_config(tmp_path)
        assert config.source_label == "fasteval.toml"
        assert list(config.profiles) == ["only"]

    def test_pyproject_without_section_gives_defaults(self, tmp_path):
        write(tmp_path / "pyproject.toml", '[project]\nname = "x"\n')
        config = find_config(tmp_path)
        assert config.source is None
        assert config.profile_names() == ["default"]
        assert config.resolve().name == "default"

    def test_no_config_files(self, tmp_path):
        config = find_config(tmp_path)
        assert config.root == tmp_path.resolve()
        assert config.resolve().paths == []


class TestResolve:
    def test_extends_merges_env_and_overrides_scalars(self, tmp_path):
        write(tmp_path / "pyproject.toml", PYPROJECT)
        ci = find_config(tmp_path).resolve("ci")
        assert ci.tags == "not slow"  # inherited
        assert ci.review_prompts is False
        assert ci.workers == "auto"
        assert ci.env == {"A": "1", "B": "ci"}
        assert ci.paths == [str(tmp_path.resolve() / "evals")]

    def test_default_profile_used_when_name_omitted(self, tmp_path):
        write(tmp_path / "pyproject.toml", PYPROJECT)
        assert find_config(tmp_path).resolve().name == "dev"

    def test_unknown_profile_suggests(self, tmp_path):
        write(tmp_path / "pyproject.toml", PYPROJECT)
        with pytest.raises(ConfigError, match="Did you mean 'dev'"):
            find_config(tmp_path).resolve("dve")

    def test_cycle_detected(self, tmp_path):
        write(
            tmp_path / "fasteval.toml",
            '[profiles.a]\nextends = "b"\n[profiles.b]\nextends = "a"\n',
        )
        with pytest.raises(ConfigError, match="cycle"):
            find_config(tmp_path).resolve("a")


class TestValidation:
    @pytest.mark.parametrize(
        "body, message",
        [
            ('tag = "x"', "unknown key 'tag'. Did you mean 'tags'"),
            ("workers = -1", "workers"),
            ('workers = "many"', "workers"),
            ("min_score = 1.5", "min_score"),
            ("min_pass_rate = true", "min_pass_rate"),
            ('paths = "evals"', "list of strings"),
            ('outputs = ["json"]', "json:path"),
            ("env = { A = 1 }", "env"),
            ('review_prompts = "no"', "true or false"),
            ("headless = 1", "true or false"),
        ],
    )
    def test_invalid_profile(self, tmp_path, body, message):
        write(tmp_path / "fasteval.toml", f"[profiles.p]\n{body}\n")
        with pytest.raises(ConfigError, match=message):
            find_config(tmp_path)

    def test_unknown_top_level_key(self, tmp_path):
        write(tmp_path / "fasteval.toml", 'profile = "x"\n')
        with pytest.raises(ConfigError, match="Did you mean 'profiles'"):
            find_config(tmp_path)

    def test_undefined_default_profile(self, tmp_path):
        write(tmp_path / "fasteval.toml", 'default_profile = "dve"\n[profiles.dev]\n')
        with pytest.raises(ConfigError, match="Did you mean 'dev'"):
            find_config(tmp_path)

    def test_invalid_toml(self, tmp_path):
        write(tmp_path / "fasteval.toml", "[profiles\n")
        with pytest.raises(ConfigError, match="Invalid TOML"):
            find_config(tmp_path)


class TestTagRegistry:
    def test_returns_tags(self, tmp_path):
        write(tmp_path / "pyproject.toml", PYPROJECT)
        assert load_tag_registry(tmp_path) == {"smoke": "Fast checks"}

    def test_broken_config_does_not_raise(self, tmp_path):
        write(tmp_path / "fasteval.toml", "[profiles\n")
        assert load_tag_registry(tmp_path) == {}
