"""Run notebooks/train_kaggle.ipynb's code cells locally on CPU in smoke mode: tiny model, no clone, no pip, no push.

Usage (.venv-train): python scripts/notebook_smoke.py [--work models/notebook-smoke]
Sets TT_SMOKE=1, TT_REPO=<this repo>, TT_WORK=<work dir> and executes every code cell in order in one namespace, so the
Kaggle run is never the first run of these cells.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(ROOT / "models" / "notebook-smoke"))
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    os.environ.update(TT_SMOKE="1", TT_REPO=str(ROOT), TT_WORK=str(work))
    nb = json.loads((ROOT / "notebooks" / "train_kaggle.ipynb").read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__notebook__"}
    for k, cell in enumerate((c for c in nb["cells"] if c["cell_type"] == "code"), 1):
        src = "".join(cell["source"])
        print(f"\n===== cell {k}: {src.splitlines()[0]} =====", flush=True)
        exec(compile(src, f"<notebook cell {k}>", "exec"), ns)  # noqa: S102 — our own notebook
    print(f"\nnotebook smoke run OK: {work / 'model'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
