"""Tests for the fasteval CLI and the plugin primitives it builds on."""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from fasteval.cli.config import find_config
from fasteval.cli.main import (
    build_parser,
    build_run_plan,
    combine_and,
    combine_or,
    main,
)

pytest_plugins = ["pytester"]

EVALS = """
import pytest
import fasteval as fe


@pytest.mark.smoke
@fe.contains()
def test_greets():
    fe.score("hello world", "hello", input="hi")


@pytest.mark.slow
@pytest.mark.parametrize("word", ["alpha", "beta"])
@fe.exact_match()
def test_echo(word):
    fe.score(word, "alpha", input=word)


def test_plain_unit():
    assert True
"""

PYPROJECT = """
[tool.pytest.ini_options]
testpaths = ["evals"]

[tool.fasteval]
default_profile = "dev"

[tool.fasteval.tags]
smoke = "Fast checks"
slow = "Slow checks"

[tool.fasteval.profiles.dev]
paths = ["evals"]
tags = "not slow"

[tool.fasteval.profiles.ci]
paths = ["evals"]
review_prompts = false
min_pass_rate = 0.9
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "test_demo.py").write_text(EVALS)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def plan_for(argv, profile=None):
    config = find_config()
    args, extra = build_parser().parse_known_args(["run", *argv])
    return build_run_plan(config, config.resolve(args.profile or profile), args, extra)


# ── expression helpers ───────────────────────────────────────────────────────


class TestCombine:
    def test_and(self):
        assert combine_and(None, "") is None
        assert combine_and("a") == "a"
        assert combine_and("a or b", "c") == "(a or b) and (c)"

    def test_or(self):
        assert combine_or([]) is None
        assert combine_or(["a"]) == "a"
        assert combine_or(["a", "b and c"]) == "(a) or (b and c)"


# ── planning (profile -> pytest args) ────────────────────────────────────────


class TestRunPlan:
    def test_default_profile(self, project):
        plan = plan_for([])
        args = plan.pytest_args
        assert args[0] == "evals"
        assert "--fe-only" in args
        assert args[args.index("-m") + 1] == "not slow"
        assert "--fe-summary" in args
        assert "--fe-output=json:.fasteval/last-run.json" in args
        assert plan.env["FASTEVAL_PROFILE"] == "dev"
        assert plan.review_prompts

    def test_cli_tags_narrow_profile_tags(self, project):
        plan = plan_for(["-t", "smoke", "-t", "rag"])
        assert plan.marker_expr == "(not slow) and ((smoke) or (rag))"

    def test_cli_paths_replace_profile_paths(self, project):
        plan = plan_for(["evals/test_demo.py"])
        assert plan.pytest_args[0] == "evals/test_demo.py"
        assert "evals" not in plan.pytest_args

    def test_ci_profile_gates_and_non_interactive(self, project):
        plan = plan_for([], profile="ci")
        assert "--no-interactive" in plan.pytest_args
        assert "--fe-min-pass-rate=0.9" in plan.pytest_args
        assert "-m" not in plan.pytest_args

    def test_ci_env_var_implies_non_interactive(self, project, monkeypatch):
        monkeypatch.setenv("CI", "true")
        assert "--no-interactive" in plan_for([]).pytest_args

    def test_passthrough_and_flags(self, project):
        plan = plan_for(["--lf", "-x", "-vv", "--tb=short", "-n", "2", "--all"])
        args = plan.pytest_args
        assert "--fe-only" not in args
        for flag in ("--lf", "-x", "-vv", "--tb=short"):
            assert flag in args
        assert args[args.index("-n") + 1] == "2"

    def test_reports(self, project):
        plan = plan_for(["--json", "r.json", "--html", "out.html"])
        assert "--fe-output=json:r.json" in plan.pytest_args
        # --html is the static UI report, written by the CLI, not a pytest flag
        assert not any("html" in a for a in plan.pytest_args)


# ── commands, end to end ─────────────────────────────────────────────────────


class TestCommands:
    def test_dry_run(self, project, capsys):
        assert main(["run", "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "-m pytest evals --fe-only -m 'not slow'" in out

    def test_run_writes_last_run_and_report(self, project, capsys):
        assert main(["run", "-q"]) == 0
        state = project / ".fasteval"
        assert (state / ".gitignore").read_text().strip().endswith("*")
        data = json.loads((state / "last-run.json").read_text())
        assert data["summary"]["total_tests"] == 1
        meta = json.loads((state / "last-run.meta.json").read_text())
        assert meta["profile"] == "dev" and meta["exit_code"] == 0

        capsys.readouterr()
        assert main(["report", "--html"]) == 0
        out = capsys.readouterr().out
        assert "1 passed" in out and "contains" in out
        assert (state / "report.html").exists()

    def test_explicit_html_path_in_headless_run(self, project, capsys):
        out = project / "out" / "r.html"
        main(["run", "-q", "--headless", "--html", str(out)])
        # The bundle may not be built in CI; either the report or a clear hint.
        captured = capsys.readouterr()
        assert out.exists() or "npm run build" in captured.err

    def test_run_failure_and_gate_exit_code(self, project):
        # test_echo[beta] fails -> pass rate 2/3 < 0.9
        assert main(["run", "-p", "ci", "-q", "--tb=no"]) == 1

    def test_list_json(self, project, capsys):
        assert main(["list", "-p", "ci", "--json"]) == 0
        items = json.loads(capsys.readouterr().out)
        names = [i["name"] for i in items]
        assert names == ["test_greets", "test_echo[alpha]", "test_echo[beta]"]
        greets = items[0]
        assert greets["tags"] == ["smoke"]
        assert greets["metrics"][0]["type"] == "contains"

    def test_list_table(self, project, capsys):
        assert main(["list"]) == 0
        out = capsys.readouterr().out
        assert "test_greets" in out and "test_echo" not in out
        assert "1 evals in 1 file" in out

    def test_list_tags(self, project, capsys):
        assert main(["list", "-p", "ci", "--tags"]) == 0
        out = capsys.readouterr().out
        assert "smoke" in out and "Fast checks" in out

    def test_profiles(self, project, capsys):
        assert main(["profiles"]) == 0
        out = capsys.readouterr().out
        assert "* dev" in out and "ci" in out
        assert main(["profiles", "ci"]) == 0
        assert "pass rate ≥ 90%" in capsys.readouterr().out

    def test_status(self, project, capsys):
        assert main([]) == 0
        assert "dev*" in capsys.readouterr().out

    def test_report_without_runs(self, project, capsys):
        assert main(["report"]) == 1
        assert "No runs yet" in capsys.readouterr().out

    def test_unknown_profile(self, project, capsys):
        assert main(["run", "-p", "nope"]) == 2
        assert "Unknown profile" in capsys.readouterr().err

    def test_unknown_args_rejected_outside_run(self, project):
        with pytest.raises(SystemExit):
            main(["list", "--bogus"])

    def test_init_appends_to_pyproject(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        (tmp_path / "evals").mkdir()
        monkeypatch.chdir(tmp_path)
        assert main(["init"]) == 0
        config = find_config()
        assert config.default_profile == "dev"
        assert config.resolve("ci").review_prompts is False
        assert config.resolve("dev").paths == [str(tmp_path.resolve() / "evals")]
        assert main(["init"]) == 0
        assert "Already configured" in capsys.readouterr().out

    def test_init_creates_fasteval_toml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert main(["init"]) == 0
        assert find_config().source_label == "fasteval.toml"
        assert find_config().resolve("ci").min_pass_rate == 0.9


# ── plugin primitives under real pytest ──────────────────────────────────────


@pytest.fixture
def evals_project(pytester):
    pytester.makepyprojecttoml(textwrap.dedent("""
            [tool.fasteval.tags]
            smoke = "Fast checks"
            slow = "Slow checks"
            """))
    pytester.makepyfile(test_demo=EVALS)
    return pytester


class TestPluginPrimitives:
    def test_fe_only_deselects_plain_tests(self, evals_project):
        result = evals_project.runpytest_subprocess(
            "--fe-only", "-p", "no:cacheprovider"
        )
        result.assert_outcomes(passed=2, failed=1, deselected=1)

    def test_fasteval_marker_added(self, evals_project):
        result = evals_project.runpytest_subprocess("-m", "fasteval and not slow")
        result.assert_outcomes(passed=1, deselected=3)

    def test_registered_tags_do_not_warn(self, evals_project):
        result = evals_project.runpytest_subprocess("--strict-markers", "-m", "smoke")
        result.assert_outcomes(passed=1, deselected=3)

    def test_gate_fails_passing_run(self, evals_project):
        # Only passing evals selected; the gate alone must flip the exit code.
        result = evals_project.runpytest_subprocess(
            "-k", "greets or alpha", "--fe-min-score=1.01"
        )
        assert result.ret == 1
        result.stdout.fnmatch_lines(["*FastEval gate failed: average score*"])

    def test_gate_passes(self, evals_project):
        result = evals_project.runpytest_subprocess(
            "-k", "greets or alpha", "--fe-min-pass-rate=1.0"
        )
        assert result.ret == 0

    def test_tag_alone_does_not_make_an_eval(self, pytester):
        pytester.makepyfile(
            test_t="import fasteval as fe\n\n@fe.tag('smoke')\ndef test_plain():\n    assert True\n"
        )
        result = pytester.runpytest_subprocess(
            "--fe-only", "-m", "smoke", "--strict-markers"
        )
        result.assert_outcomes(deselected=1)
        result = pytester.runpytest_subprocess("-m", "smoke", "--strict-markers")
        result.assert_outcomes(passed=1)

    def test_inventory(self, evals_project):
        out = evals_project.path / "inv.json"
        evals_project.runpytest_subprocess(
            "--collect-only", "--fe-only", f"--fe-inventory={out}"
        )
        items = json.loads(out.read_text())["items"]
        assert [i["function"] for i in items] == [
            "test_greets",
            "test_echo",
            "test_echo",
        ]
        assert items[1]["tags"] == ["slow"]

    @pytest.mark.skipif(
        __import__("importlib").util.find_spec("xdist") is None,
        reason="pytest-xdist not installed",
    )
    def test_xdist_results_aggregated(self, evals_project):
        report = evals_project.path / "r.json"
        result = evals_project.runpytest_subprocess(
            "-n", "2", "--fe-only", f"--fe-output=json:{report}"
        )
        result.assert_outcomes(passed=2, failed=1)
        summary = json.loads(report.read_text())["summary"]
        assert summary["total_tests"] == 3
        names = sorted(t["test_name"] for t in summary["test_summaries"])
        assert names == ["test_echo[alpha]", "test_echo[beta]", "test_greets"]


def test_python_dash_m(project):
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "fasteval", "--version"], capture_output=True, text=True
    )
    assert proc.returncode == 0 and proc.stdout.startswith("fasteval ")
