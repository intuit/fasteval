"""Tests for prompt judges, judge-call recording and run-time judge swaps."""

import asyncio
import json

import pytest

from fasteval import judges
from fasteval.core.evaluator import Evaluator
from fasteval.judges import (
    JudgeError,
    JudgeSpec,
    PromptJudgeMetric,
    infer_template,
    render_template,
    template_for_metric,
    template_variables,
)
from fasteval.models.config import MetricConfig
from fasteval.models.evaluation import EvalInput
from fasteval.utils import judge_trace


class FakeJudge:
    """Returns a fixed verdict and remembers prompts."""

    model = "fake-judge-1"
    temperature = 0.0

    def __init__(self, score=0.8, fail=False):
        self.score = score
        self.fail = fail
        self.prompts = []

    async def invoke(self, messages):
        self.prompts.append(messages[-1]["content"])
        if self.fail:
            raise RuntimeError("judge down")
        return json.dumps({"score": self.score, "reasoning": "because"})


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[tool.fasteval]\n")
    monkeypatch.chdir(tmp_path)
    judges.set_overrides({})
    yield tmp_path
    judges.set_overrides({})


SPEC = JudgeSpec(
    name="strict",
    prompt='Q: {{input}}\nA: {{ actual_output }}\nE: {{expected_output}}\nJSON {"score": 0}',
    based_on="correctness",
    threshold=0.7,
)


# ── templates ────────────────────────────────────────────────────────────────


class TestTemplates:
    def test_render_formats_each_kind(self):
        tpl = "{{input}}|{{context}}|{{history}}|{{metadata}}|{{tool_calls}}|{{nope}}"
        out = render_template(
            tpl,
            EvalInput(
                input="hi",
                context=["doc a", "doc b"],
                history=[{"role": "user", "content": "x"}],
                metadata={"k": 1},
            ),
        )
        assert out.startswith("hi|[1] doc a\n[2] doc b|user: x|")
        assert '"k": 1' in out
        assert out.endswith("|[]|{{nope}}")  # unknown names left alone

    def test_render_from_dict_and_braces_kept(self):
        assert render_template('{"score": {{input}}}', {"input": "1"}) == '{"score": 1}'

    def test_variables_in_order(self):
        assert template_variables(SPEC.prompt) == [
            "input",
            "actual_output",
            "expected_output",
        ]

    def test_template_from_builtin_llm_metric(self):
        tpl = template_for_metric("correctness", "correctness", 0.6, {})
        assert tpl and "{{actual_output}}" in tpl and "{{expected_output}}" in tpl

    def test_template_from_criteria_config(self):
        tpl = template_for_metric(
            "criteria", "tone", 0.5, {"criteria": "Is it polite?"}
        )
        assert tpl and "Is it polite?" in tpl

    @pytest.mark.parametrize(
        "metric_type", ["exact_match", "contains", "no_such_metric"]
    )
    def test_no_template_for_non_llm_metrics(self, metric_type):
        assert template_for_metric(metric_type, "m", 0.5, {}) is None

    def test_infer_template_replaces_case_values(self):
        rendered = (
            "Q: How long is the refund window?\nA: Thirty days from purchase.\nscore 4"
        )
        tpl = infer_template(
            rendered,
            {
                "input": "How long is the refund window?",
                "actual_output": "Thirty days from purchase.",
                "expected_output": "4",
            },
        )
        assert (
            tpl == "Q: {{input}}\nA: {{actual_output}}\nscore 4"
        )  # short "4" untouched


# ── specs on disk ────────────────────────────────────────────────────────────


class TestSpecs:
    def test_toml_round_trip(self, project):
        spec = JudgeSpec(
            name="tricky",
            prompt="has ''' quotes\nand \\ slashes",
            model="gpt-4o-mini",
            temperature=0.2,
        )
        path = judges.save_judge(spec)
        assert path == project / "judges" / "tricky.toml"
        loaded = judges.load_judge("tricky")
        assert loaded.prompt.strip() == spec.prompt and loaded.model == "gpt-4o-mini"
        assert [j.name for j in judges.list_judges()] == ["tricky"]

    def test_no_overwrite_unless_asked(self, project):
        judges.save_judge(SPEC)
        with pytest.raises(JudgeError, match="already exists"):
            judges.save_judge(SPEC)
        judges.save_judge(SPEC, overwrite=True)

    def test_custom_judges_dir(self, project):
        (project / "pyproject.toml").write_text(
            '[tool.fasteval]\njudges_dir = "evals/judges"\n'
        )
        assert judges.save_judge(SPEC) == project / "evals" / "judges" / "strict.toml"

    def test_missing_judge_lists_available(self, project):
        judges.save_judge(SPEC)
        with pytest.raises(JudgeError, match="available: strict"):
            judges.load_judge("other")

    @pytest.mark.parametrize(
        "data, message",
        [
            ({"name": "bad name", "prompt": "x"}, "Invalid judge name"),
            ({"name": "ok", "prompt": "  "}, "prompt is empty"),
            ({"name": "ok", "prompt": "x", "scoring": "fuzzy"}, "scoring"),
            ({"name": "ok", "prompt": "x", "threshold": 2}, "threshold"),
            ({"name": "ok", "prompt": "x", "temperature": 9}, "temperature"),
            ({"name": "ok", "prompt": "x", "color": "red"}, "Unknown judge keys"),
        ],
    )
    def test_validation(self, data, message):
        with pytest.raises(JudgeError, match=message):
            JudgeSpec.from_dict(data)

    def test_invalid_file_skipped_in_listing(self, project):
        (project / "judges").mkdir()
        (project / "judges" / "broken.toml").write_text("prompt = '''unterminated")
        judges.save_judge(SPEC)
        assert [j.name for j in judges.list_judges()] == ["strict"]

    def test_parse_override(self):
        assert judges.parse_override(" correctness = strict ") == (
            "correctness",
            "strict",
        )
        with pytest.raises(JudgeError):
            judges.parse_override("no-equals")


# ── the metric and judge-call recording ──────────────────────────────────────


class TestPromptJudgeMetric:
    def test_scores_and_records_judge(self):
        client = FakeJudge(score=0.9)
        metric = PromptJudgeMetric(SPEC, llm_client=client)
        assert metric.threshold == 0.7
        result = asyncio.run(
            Evaluator()._evaluate_metric(
                metric, EvalInput(input="q", actual_output="a", expected_output="e")
            )
        )
        assert result.passed and result.score == 0.9
        assert client.prompts == ['Q: q\nA: a\nE: e\nJSON {"score": 0}']
        assert result.details["prompt_judge"] == {
            "name": "strict",
            "based_on": "correctness",
        }
        (call,) = result.details["judge_calls"]
        assert call["judge"] == {
            "provider": "FakeJudge",
            "model": "fake-judge-1",
            "temperature": 0.0,
        }
        assert call["messages"][0]["content"] == client.prompts[0]
        assert json.loads(call["response"])["score"] == 0.9
        assert call["latency_ms"] >= 0

    def test_failed_calls_recorded_per_attempt(self):
        metric = PromptJudgeMetric(SPEC, llm_client=FakeJudge(fail=True))
        result = asyncio.run(
            Evaluator()._evaluate_metric(metric, EvalInput(actual_output="a"))
        )
        assert not result.passed
        calls = result.details["judge_calls"]
        assert len(calls) == metric.max_retries
        assert all(c["error"] == "RuntimeError: judge down" for c in calls)

    def test_loaded_by_name_via_decorator(self, project):
        judges.save_judge(SPEC)
        import fasteval as fe

        @fe.prompt_judge("strict", threshold=0.4)
        def test_fn():
            pass

        (config,) = getattr(test_fn, "_fasteval_metrics")
        metric = Evaluator()._create_metric(config)
        assert isinstance(metric, PromptJudgeMetric) and metric.threshold == 0.4

    def test_swap_keeps_eval_threshold_and_client_when_judge_omits_them(self, project):
        judges.save_judge(JudgeSpec(name="loose", prompt="{{actual_output}} score"))
        judges.set_overrides({"correctness": judges.load_judge("loose")})
        mine = FakeJudge()
        metric = Evaluator()._create_metric(
            MetricConfig(
                metric_type="correctness",
                name="correctness",
                threshold=0.85,
                llm_client=mine,
            )
        )
        assert isinstance(metric, PromptJudgeMetric)
        assert metric.threshold == 0.85 and metric._llm_client is mine

    def test_run_time_swap_replaces_metric(self, project):
        judges.save_judge(SPEC)
        judges.set_overrides({"correctness": judges.load_judge("strict")})
        metric = Evaluator()._create_metric(
            MetricConfig(metric_type="correctness", name="correctness", threshold=0.5)
        )
        assert isinstance(metric, PromptJudgeMetric)
        assert metric.name == "correctness" and metric.threshold == 0.7
        other = Evaluator()._create_metric(
            MetricConfig(metric_type="exact_match", name="exact_match")
        )
        assert not isinstance(other, PromptJudgeMetric)


class TestJudgeTrace:
    def test_pass_through_outside_capture(self):
        client = judge_trace.traced(FakeJudge())
        assert asyncio.run(client.invoke([{"role": "user", "content": "x"}]))
        assert judge_trace.traced(client) is client  # not double-wrapped
        assert judge_trace.traced(None) is None
        assert client.model == "fake-judge-1"  # attribute passthrough

    def test_sanitizes_binary_and_long_text(self, monkeypatch):
        monkeypatch.setattr(judge_trace, "MAX_TEXT_CHARS", 10)
        client = judge_trace.traced(FakeJudge())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 50},
                    {"type": "image_url", "image_url": "data:..."},
                ],
            },
            {"role": "user", "content": "y" * 50},
        ]
        with judge_trace.capture() as calls:
            asyncio.run(client.invoke(messages))
        parts = calls[0]["messages"][0]["content"]
        assert parts[0]["text"].startswith("x" * 10) and "truncated" in parts[0]["text"]
        assert parts[1] == {
            "type": "image_url",
            "omitted": "binary content not recorded",
        }
        assert "truncated" in calls[0]["messages"][1]["content"]

    def test_describe_custom_client(self):
        class Gateway:
            deployment = "internal-gpt"

            async def invoke(self, messages):
                return ""

        assert judge_trace.describe_judge(Gateway()) == {
            "provider": "Gateway",
            "model": "internal-gpt",
        }


def test_llm_alias_for_llm_client():
    """`llm=` (used in older docs) is accepted and means llm_client=."""
    import fasteval as fe

    client = FakeJudge()

    @fe.correctness(llm=client)
    def test_fn():
        pass

    (config,) = getattr(test_fn, "_fasteval_metrics")
    assert config.llm_client is client and "llm" not in config.config


# ── fe.evaluate / fe.register_metric ─────────────────────────────────────────


class TestEvaluateApi:
    def test_named_metrics_keep_decorator_defaults(self):
        import fasteval as fe

        result = fe.evaluate(
            "Paris is lovely", "Paris", input="q", metrics=["contains", "rouge"]
        )
        thresholds = {m.metric_name: m.threshold for m in result.metric_results}
        assert thresholds == {"contains": 1.0, "rouge": 0.5}

    def test_failing_result_is_returned_not_raised(self):
        import fasteval as fe

        result = fe.evaluate("nope", "Paris", metrics="exact_match")
        assert result.passed is False and result.aggregate_score == 0.0

    def test_threshold_and_judge_overrides(self):
        import fasteval as fe

        client = FakeJudge(score=0.6)
        result = fe.evaluate(
            "a", "b", metrics=["correctness"], threshold=0.7, llm_client=client
        )
        (m,) = result.metric_results
        assert m.threshold == 0.7 and not m.passed and client.prompts

    def test_async_variant(self):
        import fasteval as fe

        result = asyncio.run(fe.evaluate_async("x", "x", metrics=["exact_match"]))
        assert result.passed

    def test_accepts_metric_instances_and_configs(self):
        import fasteval as fe
        from fasteval.metrics.deterministic import ExactMatchMetric

        result = fe.evaluate(
            "x",
            "x",
            metrics=[
                ExactMatchMetric(name="em"),
                MetricConfig(metric_type="contains", name="c", threshold=1.0),
            ],
        )
        assert [m.metric_name for m in result.metric_results] == ["em", "c"]

    @pytest.mark.parametrize("metrics", [[], ["no_such_metric"], ["tag"], [42]])
    def test_bad_metrics(self, metrics):
        import fasteval as fe

        with pytest.raises((ValueError, TypeError)):
            fe.evaluate("x", metrics=metrics)


class TestRegisterMetric:
    def test_register_creates_decorator_and_name(self):
        import fasteval as fe
        from fasteval.metrics.base import Metric
        from fasteval.models.evaluation import MetricResult

        class Shouty(Metric):
            def __init__(self, char="!", **kwargs):
                super().__init__(**kwargs)
                self.char = char

            async def evaluate(self, eval_input):
                ok = (eval_input.actual_output or "").endswith(self.char)
                return MetricResult(
                    metric_name=self.name,
                    score=float(ok),
                    passed=ok,
                    threshold=self.threshold,
                )

        decorator = fe.register_metric(
            "shouty_test_metric", Shouty, default_threshold=1.0
        )
        assert getattr(fe, "shouty_test_metric") is decorator

        @decorator(char="?")
        def test_fn():
            return fe.score("really?")

        assert test_fn().passed
        assert not fe.evaluate("calm.", metrics=["shouty_test_metric"]).passed
        fe.register_metric(
            "shouty_test_metric", Shouty
        )  # re-registering our own name is fine

    @pytest.mark.parametrize("name", ["correctness", "score", "not valid", "class"])
    def test_rejects_clashes_and_bad_names(self, name):
        import fasteval as fe
        from fasteval.metrics.base import Metric

        class M(Metric):
            async def evaluate(self, eval_input):  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(ValueError):
            fe.register_metric(name, M)

    def test_rejects_non_metric_class(self):
        import fasteval as fe

        with pytest.raises(TypeError):
            fe.register_metric("not_a_metric_cls", dict)  # type: ignore[arg-type]
