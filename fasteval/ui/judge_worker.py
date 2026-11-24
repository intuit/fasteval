"""
The judge playground's worker: a warm ``pytest --fe-judge-test=-`` process.

The playground must judge with the same LLM client a real run uses, and that
client usually only exists once the tests are imported
(``fe.set_default_provider(...)`` in a test module or conftest.py). So
playground tests run in a pytest process that imported them. It's kept alive
because importing can be slow (auth, SDK clients), and restarted when the test
files change, a different set of files is needed, it dies, or it gets old
(credentials fetched at import time can expire).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import IO, Any, Dict, List, Optional, Tuple

from fasteval.testing.plugin import JUDGE_TEST_DONE

# A judge call takes seconds; a test holds at most 100 cases.
REQUEST_TIMEOUT_S = 300
# Restart after this long so credentials fetched at import time stay fresh.
MAX_AGE_S = 30 * 60
OUTPUT_TAIL = 2000


class JudgeWorkerError(Exception):
    """The worker couldn't answer; the message is shown in the playground."""


class JudgeWorker:
    def __init__(self, root: Path, env: Dict[str, str]):
        self.root = root
        self.env = env
        self.lock = threading.Lock()
        self.proc: Optional["subprocess.Popen[str]"] = None
        self.lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self.output: List[str] = []  # recent console lines, for errors
        self.key: Optional[Tuple[Any, ...]] = None
        self.started = 0.0
        self.tmp = tempfile.TemporaryDirectory(prefix="fasteval-judge-")
        self.counter = 0

    def run(
        self, paths: List[str], request: Dict[str, Any], default_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Score a playground request with the judge the tests in ``paths`` set
        up. Without ``paths`` (hand-written cases) any warm worker will do,
        else ``default_paths`` (the default profile's) are imported.
        """
        with self.lock:
            if not paths:
                paths = (
                    list(self.key[0]) if self.key and self._alive() else default_paths
                )
            key = self._key(paths)
            if (
                not self._alive()
                or key != self.key
                or time.monotonic() - self.started > MAX_AGE_S
            ):
                self._start(paths, key)
            assert self.proc is not None and self.proc.stdin is not None
            self.counter += 1
            folder = Path(self.tmp.name)
            req_path = folder / f"request-{self.counter}.json"
            out_path = folder / f"output-{self.counter}.json"
            req_path.write_text(json.dumps({**request, "output": str(out_path)}))
            try:
                self.proc.stdin.write(f"{req_path}\n")
                self.proc.stdin.flush()
            except OSError:
                pass  # it died; reported below
            self._wait_for(f"{JUDGE_TEST_DONE} {req_path}", paths)
            output: Dict[str, Any] = json.loads(out_path.read_text())
            req_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
        if "error" in output:
            if str(output["error"]).startswith("Couldn't import"):
                self.stop()  # retry the import next time (the user may fix it)
            raise JudgeWorkerError(str(output["error"])[-OUTPUT_TAIL:])
        return output

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()  # ends its request loop
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                proc.kill()

    def close(self) -> None:
        self.stop()
        self.tmp.cleanup()

    # ── internals ──

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _key(self, paths: List[str]) -> Tuple[Any, ...]:
        """Changes when the files (or any Python file under a folder) change."""
        files: List[Path] = []
        root = self.root.resolve()
        for p in paths:
            path = (self.root / p).resolve()
            if path.is_dir():
                files += [f for f in sorted(path.rglob("*.py")) if not _hidden(f, path)]
            else:
                files.append(path)
            folder = path if path.is_dir() else path.parent
            while folder == root or root in folder.parents:  # pytest's conftests
                files.append(folder / "conftest.py")
                if folder == root:
                    break
                folder = folder.parent
        files += [root / "pyproject.toml", root / "pytest.ini", root / "setup.cfg"]
        return (tuple(paths), tuple((str(f), _mtime(f)) for f in files))

    def _start(self, paths: List[str], key: Tuple[Any, ...]) -> None:
        self.stop()
        argv = [
            sys.executable, "-m", "pytest", *paths,
            "--collect-only", "-q", "-s", "-p", "no:cacheprovider",
            "--fe-judge-test=-",
        ]  # fmt: skip
        self.proc = subprocess.Popen(
            argv,
            cwd=self.root,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.key, self.started = key, time.monotonic()
        self.lines = queue.Queue()
        self.output = []
        threading.Thread(
            target=_pump, args=(self.proc.stdout, self.lines), daemon=True
        ).start()

    def _wait_for(self, marker: str, paths: List[str]) -> None:
        deadline = time.monotonic() + REQUEST_TIMEOUT_S
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.stop()
                raise JudgeWorkerError(f"Judge test took over {REQUEST_TIMEOUT_S}s")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:  # the process exited
                self.stop()
                tail = "".join(self.output)[-OUTPUT_TAIL:].strip()
                raise JudgeWorkerError(
                    "Couldn't load the tests to find the judge "
                    f"(pytest {' '.join(paths)}):\n{tail}"
                )
            if line.rstrip("\n") == marker:
                return
            self.output = [*self.output[-200:], line]


def _pump(stream: Optional[IO[str]], lines: "queue.Queue[Optional[str]]") -> None:
    if stream is not None:
        for line in stream:
            lines.put(line)
    lines.put(None)


def _hidden(path: Path, under: Path) -> bool:
    parts = path.relative_to(under).parts[:-1]
    return any(p.startswith(".") or p in ("node_modules", "__pycache__") for p in parts)


def _mtime(path: Path) -> float:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return 0.0
