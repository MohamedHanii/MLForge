"""Shared helpers for the MLForge library test notebooks.

Every notebook in this folder starts with::

    from _testkit import ROOT, SEED, Checks, load_online

Importing this module puts the repository root on ``sys.path`` so the
notebooks can ``from src.<package>._<module> import ...``.

``Checks`` records named pass / fail / skip results and prints them as they
run. ``Checks.summary()`` (the last cell of every notebook) raises
``AssertionError`` if anything failed, so executing a notebook headlessly
(``python run_all.py`` or ``jupyter nbconvert --execute``) works as a test run.
"""

from __future__ import annotations

import pathlib
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

SEED = 42


def _find_root() -> pathlib.Path:
    """Walk up from the working directory to the MLForge repository root."""
    here = pathlib.Path.cwd().resolve()
    for path in [here, *here.parents]:
        if (path / "src").is_dir() and (path / "notebooks").is_dir():
            return path
    raise FileNotFoundError("Run the notebooks from inside the MLForge repository.")


ROOT = _find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class Checks:
    """Collects named checks and prints one line per check.

    ✅ pass · ❌ fail · ⏭️ skipped (e.g. offline) · 🐞 known library bug ·
    🎉 known bug that now behaves correctly.
    """

    _ICONS = {"pass": "✅", "fail": "❌", "skip": "⏭️", "xfail": "🐞", "xpass": "🎉"}

    def __init__(self, suite: str) -> None:
        self.suite = suite
        self.results: list[dict[str, str]] = []

    def _record(self, name: str, status: str, detail: str = "") -> bool:
        self.results.append({"check": name, "status": status, "detail": detail})
        print(f"{self._ICONS[status]} {name}" + (f"  ({detail})" if detail else ""))
        return status == "pass"

    def check(self, name: str, condition: Any, detail: str = "") -> bool:
        """Pass when ``condition`` is truthy."""
        return self._record(name, "pass" if bool(condition) else "fail", detail)

    def close(self, name: str, actual: Any, expected: Any,
              rtol: float = 1e-7, atol: float = 1e-9) -> bool:
        """Pass when two arrays (or scalars) agree to within ``np.allclose`` tolerances."""
        a = np.asarray(actual, dtype=float)
        e = np.asarray(expected, dtype=float)
        if a.shape != e.shape:
            return self._record(name, "fail", f"shape {a.shape} != {e.shape}")
        err = float(np.max(np.abs(a - e))) if a.size else 0.0
        ok = np.allclose(a, e, rtol=rtol, atol=atol)
        return self._record(name, "pass" if ok else "fail", f"max |Δ| = {err:.2e}")

    def at_least(self, name: str, value: float, threshold: float) -> bool:
        """Pass when ``value >= threshold``."""
        ok = value >= threshold
        return self._record(name, "pass" if ok else "fail", f"{value:.4f} ≥ {threshold}")

    def at_most(self, name: str, value: float, threshold: float) -> bool:
        """Pass when ``value <= threshold``."""
        ok = value <= threshold
        return self._record(name, "pass" if ok else "fail", f"{value:.4f} ≤ {threshold}")

    def raises(self, name: str, exceptions: type | tuple[type, ...],
               fn: Callable[[], Any]) -> bool:
        """Pass when calling ``fn()`` raises one of ``exceptions``."""
        try:
            fn()
        except exceptions as exc:  # noqa: B030 - exceptions is a type or tuple
            return self._record(name, "pass", f"{type(exc).__name__}: {exc}"[:100])
        except Exception as exc:  # pylint: disable=broad-except
            return self._record(name, "fail", f"raised {type(exc).__name__} instead: {exc}"[:100])
        return self._record(name, "fail", "no exception raised")

    def skip(self, name: str, reason: str) -> bool:
        """Record a check that could not run (e.g. offline). Never fails the suite."""
        return self._record(name, "skip", reason)

    def known_bug(self, name: str, fn: Callable[[], Any], reason: str) -> bool:
        """Check a behaviour the library currently gets wrong (like pytest's xfail).

        ``fn()`` should return truthy when the library behaves *correctly*; an
        exception counts as incorrect. A known bug never fails the suite. Once
        it is fixed the status becomes ``xpass`` - turn it into a normal check.
        """
        try:
            correct = bool(fn())
            detail = reason
        except Exception as exc:  # pylint: disable=broad-except
            correct = False
            detail = f"{reason} | {type(exc).__name__}: {exc}"[:160]
        if correct:
            return self._record(name, "xpass", "known bug appears FIXED - make this a normal check")
        return self._record(name, "xfail", detail)

    def summary(self):
        """Print a summary table; raise ``AssertionError`` if any check failed."""
        import pandas as pd

        df = pd.DataFrame(self.results, columns=["check", "status", "detail"])
        counts = df["status"].value_counts()
        n = {s: int(counts.get(s, 0)) for s in ("pass", "fail", "skip", "xfail", "xpass")}
        print(f"{self.suite}: {n['pass']} passed, {n['fail']} failed, {n['skip']} skipped, "
              f"{n['xfail'] + n['xpass']} known bug(s) ({len(df)} checks)")
        bugs = df[df["status"].isin(["xfail", "xpass"])]
        if len(bugs):
            print("\nKnown library bugs (not counted as failures):")
            for _, row in bugs.iterrows():
                print(f"  {self._ICONS[row['status']]} {row['check']}: {row['detail']}")
        if n["fail"]:
            raise AssertionError(
                f"{n['fail']} check(s) failed in '{self.suite}':\n"
                + df[df["status"] == "fail"].to_string(index=False)
            )
        return df


def load_online(name: str, loader: Callable[[], Any],
                fallback: Callable[[], Any] | None = None):
    """Download a dataset, falling back to generated data when offline.

    Returns ``(data, online)`` where ``online`` is ``True`` if ``loader()``
    succeeded. With no ``fallback``, ``data`` is ``None`` when the download fails.
    """
    try:
        data = loader()
        print(f"🌐 {name}: downloaded / loaded from cache")
        return data, True
    except Exception as exc:  # pylint: disable=broad-except
        msg = f"⚠️ {name}: download failed ({type(exc).__name__}: {exc})"
        if fallback is None:
            print(msg + " - skipping the online checks")
            return None, False
        print(msg + " - using generated fallback data")
        return fallback(), False


@contextmanager
def timed(label: str):
    """Print how long the enclosed block took."""
    start = time.perf_counter()
    yield
    print(f"⏱️ {label}: {time.perf_counter() - start:.2f}s")
