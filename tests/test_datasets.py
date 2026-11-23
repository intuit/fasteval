"""Tests for fe.dataset / fe.case / fe.csv / fe.jsonl and fe.tag."""

import json
import textwrap

import pytest

import fasteval as fe
from fasteval.core.datasets import Dataset, parametrize_spec
from fasteval.models.evaluation import EvalInput

pytest_plugins = ["pytester"]


# ── Dataset ──────────────────────────────────────────────────────────────────


class TestDataset:
    def test_generator_function_with_ids_and_tags(self):
        @fe.dataset
        def faq():
            yield {"id": "refund", "q": "a"}
            yield fe.case(id="hours", tags=["slow"], q="b")
            yield {"q": "c", "tags": ["x"]}

        cases = faq.load()
        assert [c.id for c in cases] == ["refund", "hours", "faq-2"]
        assert cases[1].tags == ("slow",)
        assert cases[2].tags == ("x",)
        assert cases[0].values == {"q": "a"}  # id is not passed as a value
        assert list(faq) == [{"q": "a"}, {"q": "b"}, {"q": "c"}]
        assert len(faq) == 3

    def test_loads_once(self):
        calls = []

        def source():
            calls.append(1)
            return [{"q": 1}]

        ds = Dataset(source)
        ds.load()
        ds.load()
        assert calls == [1]

    def test_id_field_duplicates_and_long_ids(self):
        ds = fe.dataset(
            [{"q": "same"}, {"q": "same"}, {"q": "x" * 200}], name="d", id_field="q"
        )
        ids = [c.id for c in ds.load()]
        assert ids[:2] == ["same", "same-1"]
        assert len(ids[2]) == 60

    def test_decorator_with_options_and_iterable(self):
        ds = fe.dataset(name="named", id_field="k")(lambda: [{"k": "a"}])
        assert ds.name == "named"
        assert ds.load()[0].id == "a"

    def test_eval_input_items(self):
        ds = fe.dataset([EvalInput(input="hi", reference_id="r1")], name="d")
        case = ds.load()[0]
        assert case.id == "r1"
        assert isinstance(case.values["eval_input"], EvalInput)

    def test_rejects_bad_item(self):
        with pytest.raises(TypeError, match="expected a dict"):
            fe.dataset([42], name="d").load()

    def test_decorating_sets_attributes(self):
        ds = fe.dataset([{"a": 1}], name="d")

        @ds
        def test_x(a):
            pass

        assert test_x._fasteval_dataset is ds
        assert test_x._fasteval_data == {"type": "dataset", "name": "d"}


class TestFileDatasets:
    def test_csv(self, tmp_path):
        path = tmp_path / "qa.csv"
        path.write_text("question,expected\nWhat?,This\nWhy?,Because\n")
        ds = fe.csv(path, id_field="question")
        assert [c.id for c in ds.load()] == ["What?", "Why?"]
        assert ds.info == {"type": "csv", "path": str(path)}

    def test_jsonl(self, tmp_path):
        path = tmp_path / "g.jsonl"
        path.write_text('{"id": "a", "q": 1}\n\n{"id": "b", "q": 2}\n')
        ds = fe.jsonl(path)
        assert [c.id for c in ds.load()] == ["a", "b"]
        assert ds.info["type"] == "jsonl"

    def test_jsonl_reports_bad_line(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"q": 1}\nnot json\n')
        with pytest.raises(ValueError, match=r"bad.jsonl:2"):
            fe.jsonl(path).load()


class TestParametrizeSpec:
    def test_maps_matching_keys(self):
        ds = fe.dataset([{"question": "q", "expected": "e", "extra": 1}], name="d")

        def test_f(question, expected):
            pass

        names, rows = parametrize_spec(test_f, ds)
        assert names == ["question", "expected"]
        assert rows == [(("q", "e"), "d-0", ())]

    def test_whole_case_param(self):
        ds = fe.dataset([{"a": 1}], name="d")

        def test_f(case):
            pass

        names, rows = parametrize_spec(test_f, ds)
        assert names == ["case"]
        assert rows[0][0] == ({"a": 1},)

    def test_no_matching_keys(self):
        ds = fe.dataset([{"a": 1}], name="d")

        def test_f(b):
            pass

        with pytest.raises(ValueError, match="match none of"):
            parametrize_spec(test_f, ds)

    def test_missing_key_in_one_case(self):
        ds = fe.dataset([{"a": 1, "b": 2}, {"a": 3}], name="d")

        def test_f(a, b):
            pass

        with pytest.raises(ValueError, match="missing"):
            parametrize_spec(test_f, ds)


# ── fe.tag ───────────────────────────────────────────────────────────────────


class TestTag:
    def test_stacks_without_duplicates(self):
        @fe.tag("rag", "smoke")
        @fe.tag("smoke", "slow")
        def test_x():
            pass

        assert getattr(test_x, "_fasteval_tags") == ("smoke", "slow", "rag")

    @pytest.mark.parametrize(
        "bad", ["", "1abc", "has space", "and", "skip", "fasteval"]
    )
    def test_rejects_invalid_names(self, bad):
        with pytest.raises(ValueError):
            fe.tag(bad)

    def test_requires_a_name(self):
        with pytest.raises(ValueError, match="at least one"):
            fe.tag()


# ── end to end under pytest ──────────────────────────────────────────────────


EVALS = textwrap.dedent("""
    import pytest
    import fasteval as fe

    @fe.dataset
    def rows():
        yield {"id": "one", "word": "alpha"}
        yield fe.case(id="two", tags=["slow"], word="beta")

    @rows
    @fe.tag("rag")
    @fe.contains()
    def test_rows(word):
        fe.score(word, word[:2], input=word)

    @fe.tag("smoke")
    class TestTagged:
        @fe.exact_match()
        def test_in_class(self):
            fe.score("x", "x", input="x")
    """)


@pytest.fixture
def project(pytester):
    pytester.makepyfile(test_ds=EVALS)
    pytester.makefile(".csv", qa="word,expected\nhello,he\nworld,wo\n")
    pytester.makepyfile(test_csv=textwrap.dedent("""
            import fasteval as fe

            @fe.csv("qa.csv", id_field="word")
            @fe.contains()
            def test_csv(word, expected):
                fe.score(word, expected, input=word)
            """))
    return pytester


class TestUnderPytest:
    def test_one_case_per_row_with_ids(self, project):
        result = project.runpytest_subprocess(
            "--collect-only", "-q", "--strict-markers"
        )
        lines = set(result.stdout.lines)
        for nodeid in (
            "test_csv.py::test_csv[hello]",
            "test_csv.py::test_csv[world]",
            "test_ds.py::test_rows[one]",
            "test_ds.py::test_rows[two]",
        ):
            assert nodeid in lines

    def test_dataset_and_csv_cases_run(self, project):
        result = project.runpytest_subprocess("--strict-markers")
        result.assert_outcomes(passed=5)

    def test_per_case_tags_select(self, project):
        result = project.runpytest_subprocess("-m", "slow", "--strict-markers")
        result.assert_outcomes(passed=1, deselected=4)

    def test_class_level_tag(self, project):
        result = project.runpytest_subprocess("-m", "smoke", "--strict-markers")
        result.assert_outcomes(passed=1, deselected=4)

    def test_bad_dataset_is_a_usage_error(self, pytester):
        pytester.makepyfile(test_bad=textwrap.dedent("""
                import fasteval as fe

                @fe.dataset([{"nope": 1}], name="d")
                @fe.contains()
                def test_x(word):
                    pass
                """))
        result = pytester.runpytest_subprocess()
        assert result.ret != 0
        result.stdout.fnmatch_lines(["*match none of*"])

    def test_inventory_records_dataset(self, project):
        out = project.path / "inv.json"
        project.runpytest_subprocess(
            "--collect-only", "--fe-only", f"--fe-inventory={out}"
        )
        items = json.loads(out.read_text())["items"]
        rows = [i for i in items if i["function"] == "test_rows"]
        assert rows[0]["data"] == {"type": "dataset", "name": "rows"}
        assert rows[1]["tags"] == ["rag", "slow"]
