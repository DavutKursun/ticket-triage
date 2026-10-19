"""Draft the hard set: realistic multi-issue / typo / Turkish tickets for Davut to edit and label.

Usage: python scripts/draft_hard.py --n 300 --out data/hard/hard_draft.csv

A deterministic plan (SEED) fixes every ticket's category, urgency, sentiment, language, sentence count and
whether it carries a second issue / typos, so the set meets its quotas by construction (at n=300):
  categories equal (27-28 each) · urgency 16% urgent / 24% high / 30% medium / 30% low (48 urgent)
  sentiment 12% positive / 44% negative / 44% neutral (36 positive) · 20% Turkish (60) · 30% two issues ·
  25% with typos · 2-6 sentences.
The LLM writes the text for 10 specs per call (~30 calls, with_backoff on 429/5xx) and reports its own labels;
the CSV carries the PLANNED labels (Davut relabels every row anyway) and rows where the model disagreed are
listed at the end for review. Resumable: progress lives in <out>.progress.json (gitignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from ticket_triage.config import SEED, settings  # noqa: E402
from ticket_triage.costs import cost_usd, format_cost  # noqa: E402
from ticket_triage.data import stable_id  # noqa: E402
from ticket_triage.providers import ProviderError, complete_validated, get_provider, with_backoff  # noqa: E402
from ticket_triage.rubric import CATEGORY_DEFINITIONS, SENTIMENT_RUBRIC, URGENCY_RUBRIC  # noqa: E402
from ticket_triage.schema import CATEGORIES, Sentiment, Urgency  # noqa: E402

URGENCY_MIX = {"urgent": 0.16, "high": 0.24, "medium": 0.30, "low": 0.30}
SENTIMENT_MIX = {"positive": 0.12, "negative": 0.44, "neutral": 0.44}
LANGUAGE_MIX = {"tr": 0.20, "en": 0.80}
TWO_ISSUE_SHARE, TYPO_SHARE = 0.30, 0.25
SENTENCE_CHOICES = (2, 3, 3, 4, 4, 5, 6)
BATCH = 10
MAX_TOKENS = 16000  # 10 tickets × up to 6 sentences (Turkish is token-heavy) + the model's adaptive thinking; 6000 truncated


@dataclass
class Spec:
    i: int
    category: str
    urgency: str
    sentiment: str
    language: str
    sentences: int
    second_issue: str  # secondary category or ""
    typos: bool


class DraftTicket(BaseModel):
    i: int
    text: str  # < 40 chars filtered in code (schema constraints would be validated client-side only)
    category: str
    urgency: Urgency
    sentiment: Sentiment
    second_issue: bool
    typos: bool


class DraftBatch(BaseModel):
    tickets: list[DraftTicket] = Field(min_length=1)


SYSTEM = """You write realistic customer-support tickets for an evaluation set of a ticket classifier.
You get a numbered list of specs. For each spec write ONE ticket in the requested language whose main issue,
urgency and tone match the spec under the rubrics below, and echo the spec number as "i".

Style rules:
- Sentence count = the spec's number (2-6). Write like a real customer in a hurry: concrete details (order numbers
  such as #48213-A, dates, amounts, plan names, invented emails), greetings optional, never only polite boilerplate,
  never the same opening twice in a batch, no bullet lists.
- second_issue=yes: the message raises TWO issues. The main one (the one needing action first) decides category and
  urgency; the minor one belongs to the category named in the spec. second_issue=no: exactly one issue.
- typos=yes: 2-4 natural slips (dropped or swapped letters, missing apostrophes, lowercase "i", autocorrect words;
  in Turkish also dropped diacritics like "siparis", "odeme"), still readable. typos=no: clean spelling.
- Sentiment is TONE: negative = frustrated/angry/disappointed; positive = friendly, thankful, praising (a positive
  ticket can still be urgent); neutral = plain. Urgency is CONSEQUENCE for the customer, never tone.
- Turkish tickets: natural everyday Turkish with proper characters (ş ğ ı ö ü ç), not a translation of a template.
- Never mention the labels or the spec inside the ticket text. Vary names, products, amounts and phrasing.

Categories:
""" + "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items()) + f"""

Urgency rubric:
{URGENCY_RUBRIC}

Sentiment rubric:
{SENTIMENT_RUBRIC}

Return JSON {{"tickets": [{{"i", "text", "category", "urgency", "sentiment", "second_issue", "typos"}}, ...]}}
with exactly one ticket per spec, in the same order; category/urgency/sentiment are YOUR honest labels for the
text you wrote (they should match the spec; if you could not make them match, report what you actually wrote)."""


def _quota(n: int, shares: dict, rng: random.Random) -> list:
    """Largest-remainder apportionment of n items over the share dict, shuffled."""
    raw = {k: n * v for k, v in shares.items()}
    counts = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    out = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(out)
    return out


def make_plan(n: int, seed: int = SEED) -> list[Spec]:
    rng = random.Random(seed)
    cats = [CATEGORIES[i % len(CATEGORIES)] for i in range(n)]
    rng.shuffle(cats)
    urg, sent, lang = _quota(n, URGENCY_MIX, rng), _quota(n, SENTIMENT_MIX, rng), _quota(n, LANGUAGE_MIX, rng)
    two, typo = _quota(n, {True: TWO_ISSUE_SHARE, False: 1 - TWO_ISSUE_SHARE}, rng), _quota(n, {True: TYPO_SHARE, False: 1 - TYPO_SHARE}, rng)
    specs = []
    for i in range(n):
        second = rng.choice([c for c in CATEGORIES if c != cats[i]]) if two[i] else ""
        specs.append(Spec(i, cats[i], urg[i], sent[i], lang[i], rng.choice(SENTENCE_CHOICES), second, typo[i]))
    return specs


def _spec_line(k: int, s: Spec) -> str:
    second = f"yes (minor issue about: {s.second_issue})" if s.second_issue else "no"
    return (f"{k}. i={s.i} category={s.category}, urgency={s.urgency}, sentiment={s.sentiment}, "
            f"sentences={s.sentences}, second_issue={second}, typos={'yes' if s.typos else 'no'}")


def _user(batch: list[Spec]) -> str:
    lang = "Turkish" if batch[0].language == "tr" else "English"
    return f"Language: {lang}\nWrite one ticket per spec, same order, echo i.\n" + "\n".join(_spec_line(k + 1, s) for k, s in enumerate(batch))


def _norm(v: str) -> str:
    return v.strip().lower().replace(" ", "_").replace("-", "_")


def count_sentences(text: str) -> int:
    return len([p for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default="data/hard/hard_draft.csv")
    ap.add_argument("--rounds", type=int, default=3, help="extra passes for specs the model skipped")
    args = ap.parse_args()
    try:
        provider, model = get_provider(settings)
    except ProviderError as e:
        print("not configured:", e)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    progress = out.with_suffix(".progress.json")
    specs = make_plan(args.n)
    state = {"n": args.n, "rows": [], "tokens_in": 0, "tokens_out": 0, "calls": 0, "repaired": 0}
    if progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if saved.get("n") == args.n:
            state = saved
            print(f"resuming: {len(state['rows'])} tickets already drafted")
    done = {r["i"] for r in state["rows"]}
    seen_ids = {r["id"] for r in state["rows"]}

    for round_ in range(args.rounds):
        pending = [s for s in specs if s.i not in done]
        if not pending:
            break
        for lang in ("tr", "en"):
            group = [s for s in pending if s.language == lang]
            for b in range(0, len(group), BATCH):
                batch = group[b : b + BATCH]
                try:
                    obj, results, rep = with_backoff(partial(complete_validated, provider, SYSTEM, _user(batch), model, DraftBatch, max_tokens=MAX_TOKENS))
                except ProviderError as e:
                    print("API error (progress saved):", e)
                    progress.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                    return 4
                except Exception as e:  # noqa: BLE001 — invalid twice: skip this batch, specs come back next round
                    print(f"  batch skipped after repair: {type(e).__name__}: {str(e)[:120]}")
                    continue
                state["calls"] += len(results)
                state["repaired"] += int(rep)
                state["tokens_in"] += sum(r.input_tokens for r in results)
                state["tokens_out"] += sum(r.output_tokens for r in results)
                by_i = {s.i: s for s in batch}
                for t in obj.tickets:
                    s = by_i.get(t.i)
                    text = " ".join(t.text.split())
                    tid = stable_id(text)
                    if s is None or s.i in done or tid in seen_ids or len(text) < 40:
                        continue
                    done.add(s.i)
                    seen_ids.add(tid)
                    state["rows"].append({"i": s.i, "id": tid, "text": text, "model_category": _norm(t.category),
                                          "model_urgency": t.urgency, "model_sentiment": t.sentiment,
                                          "model_second_issue": t.second_issue, "model_typos": t.typos})
                progress.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                print(f"  round {round_ + 1} {lang}: {len(done)}/{args.n} drafted, {state['calls']} calls, "
                      f"tokens in {state['tokens_in']} / out {state['tokens_out']}, cost {format_cost(model, cost_usd(model, state['tokens_in'], state['tokens_out']))}", flush=True)

    # ---- assemble CSV with the PLANNED labels; split alternates over (language, category, id) so dev/test
    # ---- are 150/150 and balanced by language and category ----
    by_i = {s.i: s for s in specs}
    rows = []
    for r in sorted(state["rows"], key=lambda r: r["i"]):
        s = by_i[r["i"]]
        rows.append({"id": r["id"], "text": r["text"], "category": s.category, "urgency": s.urgency,
                     "sentiment": s.sentiment, "language": s.language, "split": ""})
    for k, r in enumerate(sorted(rows, key=lambda r: (r["language"], r["category"], r["id"]))):
        r["split"] = "dev" if k % 2 == 0 else "test"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "text", "category", "urgency", "sentiment", "language", "split"])
        w.writeheader()
        w.writerows(rows)
    missing = [s.i for s in specs if s.i not in done]
    print(f"\nwrote {out}: {len(rows)} rows ({len(missing)} specs unfilled{': ' + str(missing) if missing else ''}); "
          f"{state['calls']} calls, {state['repaired']} repaired, tokens in {state['tokens_in']} / out {state['tokens_out']}, "
          f"cost {format_cost(model, cost_usd(model, state['tokens_in'], state['tokens_out']))}")
    report(rows, state["rows"], by_i)
    return 0


def report(rows: list[dict], drafted: list[dict], by_i: dict[int, Spec]) -> None:
    import pandas as pd

    df = pd.DataFrame(rows)
    print("\n=== category × urgency ===")
    print(pd.crosstab(df["category"], df["urgency"], margins=True).reindex(columns=["low", "medium", "high", "urgent", "All"]))
    print("\n=== sentiment ===", df["sentiment"].value_counts().to_dict())
    print("=== language ===", df["language"].value_counts().to_dict(), "| split", df["split"].value_counts().to_dict())
    print("=== language × split ===", pd.crosstab(df["language"], df["split"]).to_dict())
    specs = [by_i[d["i"]] for d in drafted]
    print(f"=== planned: two issues {sum(bool(s.second_issue) for s in specs)}, typos {sum(s.typos for s in specs)} | "
          f"model-reported: two issues {sum(d['model_second_issue'] for d in drafted)}, typos {sum(d['model_typos'] for d in drafted)}")
    sc = pd.Series([count_sentences(r["text"]) for r in rows])
    print("=== sentences per ticket ===", sc.value_counts().sort_index().to_dict(), f"| outside 2-6: {int(((sc < 2) | (sc > 6)).sum())}")
    tr = df[df["language"] == "tr"]["text"]
    print(f"=== Turkish rows with Turkish characters: {int(tr.str.contains('[şğıöüçŞĞİÖÜÇ]').sum())}/{len(tr)}")
    mism = [(d["id"], f"{by_i[d['i']].category}/{by_i[d['i']].urgency}/{by_i[d['i']].sentiment}",
             f"{d['model_category']}/{d['model_urgency']}/{d['model_sentiment']}") for d in drafted
            if (d["model_category"], d["model_urgency"], d["model_sentiment"]) != (by_i[d["i"]].category, by_i[d["i"]].urgency, by_i[d["i"]].sentiment)]
    print(f"\n=== model disagreed with the plan on {len(mism)} rows (planned → model) ===")
    for tid, planned, got in mism:
        print(f"  {tid}  {planned} → {got}")
    rng = random.Random(SEED)
    print("\n=== 10 random rows ===")
    for r in rng.sample(rows, min(10, len(rows))):
        print(f"[{r['id']} {r['language']} {r['category']}/{r['urgency']}/{r['sentiment']} {r['split']}] {r['text']}\n")


if __name__ == "__main__":
    raise SystemExit(main())
