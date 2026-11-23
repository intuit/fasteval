"""
Datasets: generators of eval cases that parametrize test functions.

A dataset is any iterable or generator function of cases. Decorating a test
with a dataset turns every case into its own pytest test case (expanded at
collection time by the fasteval pytest plugin), so each case is reported,
re-run, filtered and parallelized individually::

    @fe.dataset
    def faq():
        yield {"question": "How do I get a refund?", "expected": "30 days"}
        yield fe.case(id="hours", tags=["slow"], question="...", expected="...")

    @faq
    @fe.correctness(threshold=0.8)
    def test_faq(question, expected):
        fe.score(agent(question), expected, input=question)

How a case reaches the test function:
- a parameter named ``case`` (or ``row``) receives the whole case dict;
- otherwise case keys matching the function's parameters are passed in
  (other keys are ignored, a missing key is an error);
- an ``EvalInput`` case is passed as ``eval_input``.
"""

from __future__ import annotations

import csv as csv_module
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from fasteval.models.evaluation import EvalInput

DATASET_ATTR = "_fasteval_dataset"
DATA_ATTR = "_fasteval_data"
_WHOLE_CASE_PARAMS = ("case", "row")
_MAX_ID = 60


@dataclass
class Case:
    """One dataset case: values for the test plus optional id and tags."""

    values: Dict[str, Any]
    id: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)


def case(id: Optional[str] = None, tags: Iterable[str] = (), **values: Any) -> Case:
    """Build a case with an explicit id and/or per-case tags."""
    return Case(values=values, id=id, tags=tuple(tags))


class Dataset:
    """A named, lazily-loaded source of cases. Use it as a test decorator."""

    def __init__(
        self,
        source: Union[Callable[[], Iterable[Any]], Iterable[Any]],
        name: Optional[str] = None,
        id_field: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._source = source
        self.name = name or getattr(source, "__name__", None) or "dataset"
        self.id_field = id_field
        self.info = info or {"type": "dataset", "name": self.name}
        self._cases: Optional[List[Case]] = None
        if callable(source):
            self.__doc__ = getattr(source, "__doc__", None)

    def __call__(self, func: Any) -> Any:
        setattr(func, DATASET_ATTR, self)
        setattr(func, DATA_ATTR, dict(self.info))
        return func

    def load(self) -> List[Case]:
        """Materialize cases once per process."""
        if self._cases is None:
            raw = self._source() if callable(self._source) else self._source
            self._cases = [self._to_case(item) for item in raw]
            self._assign_ids()
        return self._cases

    def __iter__(self) -> Any:
        return iter(c.values for c in self.load())

    def __len__(self) -> int:
        return len(self.load())

    def __repr__(self) -> str:
        return f"Dataset({self.name!r})"

    def _to_case(self, item: Any) -> Case:
        if isinstance(item, Case):
            return Case(dict(item.values), item.id, tuple(item.tags))
        if isinstance(item, EvalInput):
            return Case({"eval_input": item}, item.reference_id)
        if isinstance(item, dict):
            values = dict(item)
            case_id = values.pop("id", None) if "id" in values else None
            tags = (
                values.pop("tags", ()) if isinstance(values.get("tags"), list) else ()
            )
            return Case(values, None if case_id is None else str(case_id), tuple(tags))
        raise TypeError(
            f"Dataset {self.name!r} yielded {type(item).__name__}; "
            "expected a dict, fe.case(...) or EvalInput"
        )

    def _assign_ids(self) -> None:
        seen: Dict[str, int] = {}
        for index, c in enumerate(self._cases or []):
            base = c.id
            if base is None and self.id_field and self.id_field in c.values:
                base = str(c.values[self.id_field])
            if base is None:
                base = f"{self.name}-{index}"
            base = " ".join(str(base).split())[:_MAX_ID] or f"{self.name}-{index}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            c.id = base if count == 0 else f"{base}-{count}"


def dataset(
    source: Any = None,
    *,
    name: Optional[str] = None,
    id_field: Optional[str] = None,
) -> Any:
    """
    Declare a dataset from a generator function, function or iterable.

    Usable as ``@fe.dataset``, ``@fe.dataset(name=..., id_field=...)`` or
    ``fe.dataset([...])``. The result is a decorator for test functions.
    """
    if source is None:
        return lambda src: Dataset(src, name=name, id_field=id_field)
    return Dataset(source, name=name, id_field=id_field)


def csv_dataset(
    path: Union[str, Path], encoding: str = "utf-8", id_field: Optional[str] = None
) -> Dataset:
    filepath = Path(path)

    def rows() -> Iterable[Dict[str, Any]]:
        with open(filepath, "r", encoding=encoding, newline="") as f:
            yield from csv_module.DictReader(f)

    return Dataset(
        rows,
        name=filepath.stem,
        id_field=id_field,
        info={"type": "csv", "path": str(filepath)},
    )


def jsonl_dataset(
    path: Union[str, Path], encoding: str = "utf-8", id_field: Optional[str] = None
) -> Dataset:
    filepath = Path(path)

    def rows() -> Iterable[Dict[str, Any]]:
        with open(filepath, "r", encoding=encoding) as f:
            for number, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"{filepath}:{number}: invalid JSON: {e}"
                        ) from e

    return Dataset(
        rows,
        name=filepath.stem,
        id_field=id_field,
        info={"type": "jsonl", "path": str(filepath)},
    )


def parametrize_spec(
    func: Any, ds: Dataset
) -> Tuple[List[str], List[Tuple[Tuple[Any, ...], str, Tuple[str, ...]]]]:
    """
    (argnames, [(values, id, tags), ...]) for pytest parametrization.
    Raises ValueError with a helpful message on a shape mismatch.
    """
    cases = ds.load()
    params = inspect.signature(func).parameters
    whole = next((p for p in _WHOLE_CASE_PARAMS if p in params), None)
    if whole:
        return [whole], [((c.values,), c.id or "", c.tags) for c in cases]

    keys: List[str] = []
    for c in cases:
        for k in c.values:
            if k in params and k not in keys:
                keys.append(k)
    if cases and not keys:
        raise ValueError(
            f"Dataset {ds.name!r} keys {sorted(cases[0].values)} match none of "
            f"{func.__name__}() parameters {list(params)}. Add a parameter named "
            "after a key, or a 'case' parameter to receive the whole case."
        )
    rows = []
    for c in cases:
        missing = [k for k in keys if k not in c.values]
        if missing:
            raise ValueError(f"Dataset {ds.name!r} case {c.id!r} is missing {missing}")
        rows.append((tuple(c.values[k] for k in keys), c.id or "", c.tags))
    return keys, rows
