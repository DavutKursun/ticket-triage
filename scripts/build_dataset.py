"""Download Bitext from the Hugging Face Hub, dedupe, stratified 80/10/10 split → data/splits/*.jsonl.

Usage: python scripts/build_dataset.py [--max-per-category N]
The dataset is CDLA-Sharing-1.0: data/splits/ATTRIBUTION.md is written alongside.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ticket_triage.config import HF_DATASET, SEED, SPLITS  # noqa: E402
from ticket_triage.data import dedupe_and_split, write_jsonl  # noqa: E402
from ticket_triage.schema import CATEGORIES  # noqa: E402

ATTRIBUTION = f"""# Data attribution

`data/splits/*.jsonl` are derived from **{HF_DATASET}** (Bitext),
licensed under CDLA-Sharing-1.0. Derived label files in `data/silver/` and `data/handcheck/`
are shared under the same license. `data/hard/hard.csv` is original work (MIT, like the code).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-category", type=int, default=0, help="cap rows per category (0 = all)")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    df = ds.to_pandas()
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("instruction") or cols.get("text")
    df = pd.DataFrame({"text": df[text_col], "category": df[cols["category"]].str.lower(), "intent": df[cols["intent"]].str.lower()})
    found = sorted(df["category"].unique())
    if tuple(found) != tuple(sorted(CATEGORIES)):
        print("Dataset categories differ from schema.CATEGORIES:")
        print("  dataset:", found)
        print("  schema :", sorted(CATEGORIES))
        return 2
    if args.max_per_category:
        df = df.groupby("category", group_keys=False).apply(lambda g: g.sample(min(len(g), args.max_per_category), random_state=SEED))
    splits = dedupe_and_split(df, seed=SEED)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        write_jsonl(SPLITS / f"{name}.jsonl", part.to_dict("records"))
        print(f"{name:5s} {len(part):6d} rows")
    (SPLITS / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    print(pd.concat([p.assign(split=n) for n, p in splits.items()]).groupby(["category", "split"]).size().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
