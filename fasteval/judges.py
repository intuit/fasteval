"""
Prompt judges: LLM judges defined by an editable prompt template.

A judge is a TOML file in the project's judges directory (``judges/`` by
default, see ``[tool.fasteval] judges_dir``)::

    # judges/refund_correctness.toml
    name = "refund_correctness"
    description = "Stricter correctness for refund answers"
    based_on = "correctness"
    model = "gpt-4o-mini"        # optional; omit to use the default provider
    temperature = 0.0
    threshold = 0.7              # optional; omit to keep the eval's threshold
    scoring = "continuous"       # or "binary"
    prompt = '''
    Grade the OUTPUT against the EXPECTED answer...
    INPUT: {{input}}
    OUTPUT: {{actual_output}}
    EXPECTED: {{expected_output}}
    Return JSON: {"score": <0.0-1.0>, "reasoning": "<why>"}
    '''

Use a judge in code with ``@fe.prompt_judge("refund_correctness")``, or
swap it in for an existing metric at run time without touching code::

    fasteval run --judge correctness=refund_correctness

Templates use ``{{variable}}`` placeholders; other braces (e.g. JSON
examples) are left alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fasteval.metrics.llm import BaseLLMMetric
from fasteval.models.evaluation import EvalInput

DEFAULT_JUDGES_DIR = "judges"

TEMPLATE_VARS = (
    "input",
    "actual_output",
    "expected_output",
    "context",
    "retrieval_context",
    "history",
    "tool_calls",
    "expected_tools",
    "metadata",
)

_VAR_RE = re.compile(r"\{\{\s*(" + "|".join(TEMPLATE_VARS) + r")\s*\}\}")
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\-]{0,63}$")


class JudgeError(ValueError):
    """Invalid judge definition or reference."""


# ── templates ────────────────────────────────────────────────────────────────


def template_variables(template: str) -> List[str]:
    """Variables used by a template, in first-use order."""
    seen: List[str] = []
    for match in _VAR_RE.finditer(template):
        if match.group(1) not in seen:
            seen.append(match.group(1))
    return seen


def _format_value(name: str, value: Any) -> str:
    if value is None:
        return ""
    if name in ("context", "retrieval_context") and isinstance(value, list):
        return "\n".join(f"[{i + 1}] {v}" for i, v in enumerate(value))
    if name == "history" and isinstance(value, list):
        lines = []
        for turn in value:
            if isinstance(turn, dict):
                lines.append(f"{turn.get('role', 'user')}: {turn.get('content', '')}")
            else:
                lines.append(str(turn))
        return "\n".join(lines)
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)


def render_template(template: str, eval_input: Any) -> str:
    """Fill ``{{variable}}`` placeholders from an EvalInput (or dict)."""
    if isinstance(eval_input, EvalInput):
        data = eval_input.model_dump(mode="json")
    else:
        data = dict(eval_input or {})
    return _VAR_RE.sub(
        lambda m: _format_value(m.group(1), data.get(m.group(1))), template
    )


def template_for_metric(
    metric_type: str, name: str, threshold: float, config: Optional[Dict[str, Any]]
) -> Optional[str]:
    """
    Derive an editable template from a built-in LLM metric by rendering its
    prompt with placeholder inputs. Returns None for non-LLM metrics.
    """
    from fasteval.core.evaluator import Evaluator
    from fasteval.models.config import MetricConfig

    try:
        metric = Evaluator()._create_metric(
            MetricConfig(
                metric_type=metric_type,
                name=name,
                threshold=threshold,
                config=dict(config or {}),
            )
        )
    except Exception:
        return None
    if not isinstance(metric, BaseLLMMetric) or type(metric).get_evaluation_prompt is (
        BaseLLMMetric.get_evaluation_prompt
    ):
        return None
    if isinstance(metric, PromptJudgeMetric):
        return metric.spec.prompt
    sentinel = EvalInput(
        input="{{input}}",
        actual_output="{{actual_output}}",
        expected_output="{{expected_output}}",
        context=["{{context}}"],
        retrieval_context=["{{retrieval_context}}"],
        history=[{"role": "user", "content": "{{history}}"}],
    )
    try:
        return metric.get_evaluation_prompt(sentinel)
    except Exception:
        return None


def infer_template(rendered_prompt: str, eval_input: Dict[str, Any]) -> str:
    """
    Fallback: turn a rendered prompt back into a template by replacing the
    case's own values with placeholders. Short values are left alone to
    avoid replacing incidental matches.
    """
    replacements = []
    for var in ("actual_output", "expected_output", "input"):
        value = eval_input.get(var)
        if isinstance(value, str) and len(value.strip()) >= 8:
            replacements.append((value, "{{" + var + "}}"))
    for var in ("context", "retrieval_context"):
        values = eval_input.get(var) or []
        for doc in values:
            if isinstance(doc, str) and len(doc.strip()) >= 8:
                replacements.append((doc, "{{" + var + "}}"))
    template = rendered_prompt
    for value, token in sorted(replacements, key=lambda r: -len(r[0])):
        if token in template and token.endswith("context}}"):
            template = template.replace(value, "")
        else:
            template = template.replace(value, token)
    return template


# ── judge specs (files) ──────────────────────────────────────────────────────


@dataclass
class JudgeSpec:
    name: str
    prompt: str
    description: str = ""
    based_on: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    threshold: Optional[float] = None
    scoring: Literal["continuous", "binary"] = "continuous"

    def validate(self) -> "JudgeSpec":
        if not _NAME_RE.match(self.name or ""):
            raise JudgeError(
                f"Invalid judge name {self.name!r}: use letters, digits, '_' or '-'"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise JudgeError("Judge prompt is empty")
        if self.scoring not in ("continuous", "binary"):
            raise JudgeError("scoring must be 'continuous' or 'binary'")
        for key in ("threshold",):
            value = getattr(self, key)
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise JudgeError(f"{key} must be between 0 and 1")
        if self.temperature is not None and not (0.0 <= float(self.temperature) <= 2.0):
            raise JudgeError("temperature must be between 0 and 2")
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeSpec":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise JudgeError(f"Unknown judge keys: {', '.join(sorted(unknown))}")
        spec = cls(**{k: v for k, v in data.items() if v is not None and v != ""})
        return spec.validate()

    def to_toml(self) -> str:
        lines = []
        for key, value in asdict(self).items():
            if key == "prompt" or value is None or value == "":
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append(f"prompt = {_toml_multiline(self.prompt)}")
        return "\n".join(lines) + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    return json.dumps(str(value))  # JSON string escaping is valid TOML


def _toml_multiline(text: str) -> str:
    if "'''" not in text:
        return "'''\n" + text.rstrip("\n") + "\n'''"
    escaped = text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return '"""\n' + escaped.rstrip("\n") + '\n"""'


def judges_dir(root: Optional[Path] = None) -> Path:
    from fasteval.cli.config import find_config

    config = find_config(root)
    return config.root / (config.judges_dir or DEFAULT_JUDGES_DIR)


def list_judges(root: Optional[Path] = None) -> List[JudgeSpec]:
    directory = judges_dir(root)
    if not directory.is_dir():
        return []
    specs = []
    for path in sorted(directory.glob("*.toml")):
        try:
            specs.append(load_judge_file(path))
        except JudgeError:
            continue
    return specs


def load_judge_file(path: Path) -> JudgeSpec:
    from fasteval.cli.config import ConfigError, _read_toml

    try:
        data = _read_toml(path)
    except ConfigError as e:
        raise JudgeError(str(e)) from e
    data.setdefault("name", path.stem)
    return JudgeSpec.from_dict(data)


def load_judge(name: str, root: Optional[Path] = None) -> JudgeSpec:
    if not _NAME_RE.match(name or ""):
        raise JudgeError(f"Invalid judge name {name!r}")
    path = judges_dir(root) / f"{name}.toml"
    if not path.is_file():
        available = ", ".join(s.name for s in list_judges(root)) or "none"
        raise JudgeError(
            f"Judge {name!r} not found in {path.parent} (available: {available})"
        )
    return load_judge_file(path)


def save_judge(
    spec: JudgeSpec, root: Optional[Path] = None, overwrite: bool = False
) -> Path:
    spec.validate()
    directory = judges_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.name}.toml"
    if path.exists() and not overwrite:
        raise JudgeError(f"Judge {spec.name!r} already exists")
    path.write_text(spec.to_toml(), encoding="utf-8")
    return path


# ── the metric ───────────────────────────────────────────────────────────────


def load_judge_client(spec: str, root: Path) -> Any:
    """
    Import ``[tool.fasteval] judge_client = "package.module:attribute"``.

    ``root`` (the project root) is put on ``sys.path`` so project-local
    modules resolve. A class is instantiated; any other object is used as is.
    """
    import importlib
    import sys

    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise JudgeError(
            f"judge_client {spec!r} must look like 'package.module:attribute'"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        obj: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as e:
        raise JudgeError(f"Can't load judge_client {spec!r}: {e}") from e
    client = obj() if isinstance(obj, type) else obj
    if not callable(getattr(client, "invoke", None)):
        raise JudgeError(f"judge_client {spec!r} has no invoke(messages) method")
    return client


def client_for_spec(spec: JudgeSpec, fallback: Any = None) -> Any:
    """LLM client for a judge: its model if set, else ``fallback``/default."""
    if spec.model:
        from fasteval.providers.registry import create_provider_for_model

        client = create_provider_for_model(spec.model)
        if spec.temperature is not None and hasattr(client, "temperature"):
            client.temperature = spec.temperature
        return client
    return fallback


class PromptJudgeMetric(BaseLLMMetric):
    """LLM metric whose prompt is a user-editable template (a JudgeSpec)."""

    def __init__(
        self,
        judge: Any = None,
        name: Optional[str] = None,
        threshold: Optional[float] = None,
        weight: float = 1.0,
        llm_client: Any = None,
        **_: Any,
    ) -> None:
        spec = judge if isinstance(judge, JudgeSpec) else load_judge(str(judge))
        self.spec = spec
        super().__init__(
            name=name or spec.name,
            threshold=(
                threshold
                if threshold is not None
                else (spec.threshold if spec.threshold is not None else 0.5)
            ),
            weight=weight,
            llm_client=llm_client or client_for_spec(spec),
            scoring_type=spec.scoring,
        )

    def get_evaluation_prompt(self, eval_input: EvalInput) -> str:
        return render_template(self.spec.prompt, eval_input)

    async def evaluate(self, eval_input: EvalInput) -> Any:
        result = await super().evaluate(eval_input)
        result.details = {
            **result.details,
            "prompt_judge": {"name": self.spec.name, "based_on": self.spec.based_on},
        }
        return result


# ── run-time judge swaps (fasteval run --judge metric=judge) ────────────────

_overrides: Dict[str, JudgeSpec] = {}


def parse_override(text: str) -> tuple[str, str]:
    metric, sep, judge = text.partition("=")
    if not sep or not metric.strip() or not judge.strip():
        raise JudgeError(f"Invalid --judge {text!r}: expected METRIC=JUDGE")
    return metric.strip(), judge.strip()


def set_overrides(overrides: Dict[str, JudgeSpec]) -> None:
    _overrides.clear()
    _overrides.update(overrides)


def get_override(metric_name: str) -> Optional[JudgeSpec]:
    return _overrides.get(metric_name)
