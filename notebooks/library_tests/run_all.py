"""Execute the MLForge library test notebooks and report the results.

Usage (from anywhere inside the repository)::

    python notebooks/library_tests/run_all.py              # run all, save outputs into the notebooks
    python notebooks/library_tests/run_all.py 07 08        # only notebooks whose name starts with 07 / 08
    python notebooks/library_tests/run_all.py --no-save    # run without overwriting the notebooks

Exits with status 1 if any notebook has a failing check or crashes, so it can
be used in CI. Known library bugs (🐞) and offline skips (⏭️) do not fail a run.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = pathlib.Path(__file__).resolve().parent
SUMMARY = re.compile(r"^\d\d [^:]+: (\d+) passed, (\d+) failed, (\d+) skipped, (\d+) known bug")


def run(path: pathlib.Path, save: bool, timeout: int) -> dict:
    """Execute one notebook and return its parsed summary."""
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(HERE)}})
    start = time.perf_counter()
    error = None
    try:
        client.execute()
    except CellExecutionError as exc:
        error = str(exc).strip().splitlines()[-1][:120]
    elapsed = time.perf_counter() - start
    if save:
        nbformat.write(nb, path)

    counts = None
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            for line in out.get("text", "").splitlines():
                if match := SUMMARY.match(line):
                    counts = tuple(int(g) for g in match.groups())
    passed, failed, skipped, bugs = counts or (0, 0, 0, 0)
    ok = error is None and counts is not None and failed == 0
    return {"name": path.stem, "ok": ok, "passed": passed, "failed": failed, "skipped": skipped,
            "bugs": bugs, "seconds": elapsed, "error": error}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefixes", nargs="*", help="only run notebooks whose file name starts with one of these")
    parser.add_argument("--no-save", action="store_true", help="do not write executed outputs back to the notebooks")
    parser.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds (default 900)")
    args = parser.parse_args()

    notebooks = sorted(p for p in HERE.glob("[0-9][0-9]_*.ipynb")
                       if not args.prefixes or p.name.startswith(tuple(args.prefixes)))
    if not notebooks:
        print("No notebooks matched.")
        return 1

    results = []
    for path in notebooks:
        print(f"▶ {path.name} ...", end=" ", flush=True)
        r = run(path, save=not args.no_save, timeout=args.timeout)
        results.append(r)
        print(f"{'PASS' if r['ok'] else 'FAIL'} ({r['seconds']:.0f}s)")

    print(f"\n{'notebook':28s} {'result':6s} {'passed':>6s} {'failed':>6s} {'skipped':>7s} {'bugs':>4s} {'time':>6s}")
    for r in results:
        print(f"{r['name']:28s} {'PASS' if r['ok'] else 'FAIL':6s} {r['passed']:6d} {r['failed']:6d} "
              f"{r['skipped']:7d} {r['bugs']:4d} {r['seconds']:5.0f}s")
        if r["error"] and r["failed"] == 0:
            print(f"    error: {r['error']}")
    total = {k: sum(r[k] for r in results) for k in ("passed", "failed", "skipped", "bugs")}
    print(f"\nTotal: {total['passed']} passed, {total['failed']} failed, {total['skipped']} skipped, "
          f"{total['bugs']} known library bug(s)")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
