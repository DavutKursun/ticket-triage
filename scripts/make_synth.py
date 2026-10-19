"""Synthetic TRAIN-ONLY rows, written by the LLM from the rubric and verified blind by the silver labeller.

Usage:
  python scripts/make_synth.py                      # plan "minority" (M3-1): 120 urgent, 120 high, 120 positive
  python scripts/make_synth.py --plan control       # plan "control" (M3-2): 240 long calm tickets
Options: --per-class N (minority) · --scale X (control) · --seed S · --delay s · --max-usd $ · --out path

Plans (deterministic from --seed):
- minority: the classes silver train lacks. urgent/high rows get a negative or neutral tone; positive rows get low or
  medium urgency. ~30% Turkish, 2-5 sentences, ~25% with typos, categories round-robin over all 11.
- control: the SAME style, length and language mix, but the ordinary label mix (80 low/neutral, 40 low/negative,
  60 medium/neutral, 60 medium/negative). It removes the shortcut "Turkish or long => urgent/positive": without it every
  Turkish and every >16-word train row is an urgent/high/positive one, which is exactly the hard set's profile.
Haiku writes 10 messages per call; then the regular silver labeller (SILVER_SYSTEM / SILVER_INSTRUCTION / SilverLabel)
labels every message blind and a row is kept only when BOTH labels equal the constructed ones. Kept rows are merged into
<out> (source="synthetic", plan, seed, target_class), rejected rows into <out stem>.rejected.jsonl; rows of other plans
are preserved. Train only: never part of dev, test or the hard set. Resumable via a per-plan progress file (gitignored).
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from ticket_triage.config import SEED, settings  # noqa: E402
from ticket_triage.costs import cost_usd, format_cost  # noqa: E402
from ticket_triage.data import read_jsonl, stable_id  # noqa: E402
from ticket_triage.providers import ProviderError, complete_validated, get_provider, with_backoff  # noqa: E402
from ticket_triage.rubric import CATEGORY_DEFINITIONS, PROMPT_VERSION, SENTIMENT_RUBRIC, SILVER_INSTRUCTION, SILVER_SYSTEM, URGENCY_RUBRIC  # noqa: E402
from ticket_triage.schema import CATEGORIES, Sentiment, SilverLabel, Urgency  # noqa: E402

# minority plan: class -> (fixed task, fixed value, the other label's mix)
CLASSES = {
    "urgent": ("urgency", "urgent", {"negative": 0.7, "neutral": 0.3}),
    "high": ("urgency", "high", {"negative": 0.5, "neutral": 0.5}),
    "positive": ("sentiment", "positive", {"low": 0.6, "medium": 0.4}),
}
# control plan: (urgency, sentiment, count) — the ordinary label mix at the synthetic rows' length and language mix
CONTROL_CELLS = (("low", "neutral", 80), ("low", "negative", 40), ("medium", "neutral", 60), ("medium", "negative", 60))
CONTROL_NOTE = ("These are calm-consequence tickets: urgency stays low or medium even when the customer is annoyed. Write them as "
                "long and detailed as the spec's sentence count asks, with concrete details (order numbers, dates, amounts, plan "
                "names) like any real ticket, but nothing is blocked, no money is being lost right now and there is no deadline today.")
TURKISH_SHARE, TYPO_SHARE = 0.30, 0.25
SENTENCE_CHOICES = (2, 3, 3, 4, 5)
BATCH = 10
GEN_MAX_TOKENS = 16000
LEGACY_PROGRESS = "train_synth.progress.json"  # the M3-1 minority run (seed 7)


@dataclass
class Spec:
    i: int
    cls: str
    category: str
    urgency: str
    sentiment: str
    language: str
    sentences: int
    typos: bool


class SynthTicket(BaseModel):
    i: int
    text: str
    urgency: Urgency
    sentiment: Sentiment


class SynthBatch(BaseModel):
    tickets: list[SynthTicket] = Field(min_length=1)


GEN_SYSTEM = """You write realistic customer-support messages for training a ticket classifier. You get numbered specs;
for each one write ONE message in the requested language, from a customer to a company's support team, whose
main issue belongs to the spec's category and whose urgency and tone match the spec EXACTLY under the rubrics below.

Make the label unmistakable from the text alone, because a second reader will label it blind:
- urgent: say explicitly what is happening right now or today — money leaving the account, a duplicate or
  unauthorised charge, a service that is down, a hard deadline today — and that it needs action within hours.
- high: the customer is clearly blocked (cannot log in, cannot pay, order stuck) but no money is actively being lost.
- medium / low: an inconvenience that can wait a few days, or a plain question with no time pressure.
- positive: a thankful, friendly or appreciative tone throughout (praise for the company, "thanks a lot", warmth),
  even while asking for something. negative: frustrated, angry or disappointed. neutral: plain, no emotion.
Style: the spec's sentence count (2-5); concrete details (order numbers like #48213-A, amounts, dates, plan names);
greetings optional; never the same opening twice; no bullet lists; never mention labels or the spec.
typos=yes: 2-3 natural slips (dropped or swapped letters, missing apostrophes, lowercase "i"; in Turkish also dropped
diacritics such as "siparis", "odeme"), still readable. typos=no: clean spelling.
Turkish messages: natural everyday Turkish with proper characters (ş ğ ı ö ü ç), not translated templates.

Categories:
""" + "\n".join(f"- {k}: {v}" for k, v in CATEGORY_DEFINITIONS.items()) + f"""

Urgency rubric:
{URGENCY_RUBRIC}

Sentiment rubric:
{SENTIMENT_RUBRIC}

Return JSON {{"tickets": [{{"i", "text", "urgency", "sentiment"}}, ...]}}: one ticket per spec, same order, echo i;
urgency/sentiment are your honest labels for the text you wrote (they should equal the spec)."""


def _quota(n: int, shares: dict, rng: random.Random) -> list:
    raw = {k: n * v for k, v in shares.items()}
    counts = {k: int(x) for k, x in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    out = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(out)
    return out


def make_plan(per_class: int, seed: int) -> list[Spec]:
    """Plan "minority" (M3-1): unchanged so the committed rows stay reproducible."""
    rng = random.Random(seed)
    specs: list[Spec] = []
    for cls, (fixed_task, fixed_value, other_mix) in CLASSES.items():
        cats = [CATEGORIES[k % len(CATEGORIES)] for k in range(per_class)]
        rng.shuffle(cats)
        other = _quota(per_class, other_mix, rng)
        lang = _quota(per_class, {"tr": TURKISH_SHARE, "en": 1 - TURKISH_SHARE}, rng)
        typo = _quota(per_class, {True: TYPO_SHARE, False: 1 - TYPO_SHARE}, rng)
        for k in range(per_class):
            urg, sent = (fixed_value, other[k]) if fixed_task == "urgency" else (other[k], fixed_value)
            specs.append(Spec(len(specs), cls, cats[k], urg, sent, lang[k], rng.choice(SENTENCE_CHOICES), typo[k]))
    return specs


def make_control_plan(seed: int, scale: float = 1.0) -> list[Spec]:
    """Plan "control": per label cell, categories round-robin from a random offset and language/typo quotas, so every
    cell has the same ~30% Turkish and length mix as the minority rows."""
    rng = random.Random(seed)
    specs: list[Spec] = []
    for urg, sent, n0 in CONTROL_CELLS:
        n = max(1, round(n0 * scale))
        off = rng.randrange(len(CATEGORIES))
        cats = [CATEGORIES[(off + k) % len(CATEGORIES)] for k in range(n)]
        rng.shuffle(cats)
        lang = _quota(n, {"tr": TURKISH_SHARE, "en": 1 - TURKISH_SHARE}, rng)
        typo = _quota(n, {True: TYPO_SHARE, False: 1 - TYPO_SHARE}, rng)
        for k in range(n):
            specs.append(Spec(len(specs), "control", cats[k], urg, sent, lang[k], rng.choice(SENTENCE_CHOICES), typo[k]))
    return specs


def _user(batch: list[Spec], note: str = "") -> str:
    lang = "Turkish" if batch[0].language == "tr" else "English"
    lines = [f"{k + 1}. i={s.i} category={s.category}, urgency={s.urgency}, sentiment={s.sentiment}, sentences={s.sentences}, "
             f"typos={'yes' if s.typos else 'no'}" for k, s in enumerate(batch)]
    head = f"Language: {lang}\n" + (note + "\n" if note else "") + "Write one message per spec, same order, echo i.\n"
    return head + "\n".join(lines)


def _merge(path: Path, new_rows: list[dict], plan: str, seed: int) -> int:
    """Replace this (plan, seed)'s rows in path, keep every other plan's rows. Rows written before plans existed count
    as ("minority", 7)."""
    keep = [r for r in read_jsonl(path) if (r.get("plan", "minority"), r.get("seed", 7)) != (plan, seed)]
    with path.open("w", encoding="utf-8") as fh:
        for r in keep + new_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", choices=["minority", "control"], default="minority")
    ap.add_argument("--per-class", type=int, default=120, help="minority plan: messages per class")
    ap.add_argument("--scale", type=float, default=1.0, help="control plan: multiply the 80/40/60/60 cell counts")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--delay", type=float, default=0.3, help="seconds between verification calls")
    ap.add_argument("--max-usd", type=float, default=0.15, help="stop starting new calls once this plan has spent this much")
    ap.add_argument("--out", default="data/silver/train_synth.jsonl")
    ap.add_argument("--rounds", type=int, default=3, help="extra generation passes for specs the model skipped")
    args = ap.parse_args()
    try:
        provider, model = get_provider(settings)
    except ProviderError as e:
        print("not configured:", e)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = out.with_name(out.stem + ".rejected.jsonl")
    if args.plan == "minority":
        specs, size_key, note = make_plan(args.per_class, args.seed), args.per_class, ""
        legacy = args.seed == SEED and out.name == "train_synth.jsonl"
        progress = out.with_name(LEGACY_PROGRESS) if legacy else out.with_name(f"{out.stem}.minority.s{args.seed}.progress.json")
    else:
        specs, size_key, note = make_control_plan(args.seed, args.scale), args.scale, CONTROL_NOTE
        progress = out.with_name(f"{out.stem}.control.s{args.seed}.progress.json")
    by_i = {s.i: s for s in specs}
    state = {"plan": args.plan, "size": size_key, "seed": args.seed, "gen": {}, "verify": {},
             "tokens": {"gen_in": 0, "gen_out": 0, "ver_in": 0, "ver_out": 0}, "calls": 0, "repaired": 0}
    if progress.exists():
        saved = json.loads(progress.read_text(encoding="utf-8"))
        same = (saved.get("plan", "minority"), saved.get("size", saved.get("per_class")), saved.get("seed")) == (args.plan, size_key, args.seed)
        if same:
            state.update(saved)
            print(f"resuming {args.plan}: {len(state['gen'])} generated, {len(state['verify'])} verified")

    def save() -> None:
        progress.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    def spent() -> float:
        t = state["tokens"]
        return cost_usd(model, t["gen_in"] + t["ver_in"], t["gen_out"] + t["ver_out"])

    def usage() -> str:
        t = state["tokens"]
        return f"gen {t['gen_in']}/{t['gen_out']} + verify {t['ver_in']}/{t['ver_out']} tokens, cost {format_cost(model, spent())}"

    over_budget = False
    # ---- 1. generation, 10 specs per call, grouped by language ----
    seen_texts = {g["text"] for g in state["gen"].values()}
    for round_ in range(args.rounds):
        pending = [s for s in specs if str(s.i) not in state["gen"]]
        if not pending or over_budget:
            break
        for lang in ("tr", "en"):
            group = [s for s in pending if s.language == lang]
            for b in range(0, len(group), BATCH):
                if spent() >= args.max_usd:
                    print(f"budget ${args.max_usd} reached during generation — stopping (progress saved)")
                    over_budget = True
                    break
                batch = group[b : b + BATCH]
                try:
                    obj, results, rep = with_backoff(partial(complete_validated, provider, GEN_SYSTEM, _user(batch, note), model, SynthBatch, max_tokens=GEN_MAX_TOKENS))
                except ProviderError as e:
                    print("API error (progress saved):", e)
                    save()
                    return 4
                except Exception as e:  # noqa: BLE001 — invalid twice: specs come back next round
                    print(f"  batch skipped: {type(e).__name__}: {str(e)[:120]}")
                    continue
                state["calls"] += len(results)
                state["repaired"] += int(rep)
                state["tokens"]["gen_in"] += sum(r.input_tokens for r in results)
                state["tokens"]["gen_out"] += sum(r.output_tokens for r in results)
                want = {s.i for s in batch}
                for t in obj.tickets:
                    text = " ".join(t.text.split())
                    if t.i not in want or str(t.i) in state["gen"] or len(text) < 40 or text in seen_texts:
                        continue
                    seen_texts.add(text)
                    state["gen"][str(t.i)] = {"text": text, "model_urgency": t.urgency, "model_sentiment": t.sentiment}
                save()
                print(f"  gen round {round_ + 1} {lang}: {len(state['gen'])}/{len(specs)} generated | {usage()}", flush=True)
            if over_budget:
                break

    # ---- 2. blind verification with the production silver labeller ----
    todo = [i for i in state["gen"] if i not in state["verify"]]
    print(f"verifying {len(todo)} generated messages with the silver labeller (prompt {PROMPT_VERSION})", flush=True)
    for n, i in enumerate(todo, 1):
        if spent() >= args.max_usd:
            print(f"budget ${args.max_usd} reached during verification — unverified rows are left out (progress saved)")
            break
        s, g = by_i[int(i)], state["gen"][i]
        user = SILVER_INSTRUCTION.format(category=s.category, text=g["text"])
        try:
            obj, results, rep = with_backoff(partial(complete_validated, provider, SILVER_SYSTEM, user, model, SilverLabel))
        except ProviderError as e:
            print("API error (progress saved):", e)
            save()
            return 4
        except Exception as e:  # noqa: BLE001
            print(f"  verify skipped for spec {i}: {type(e).__name__}: {str(e)[:100]}")
            state["verify"][i] = {"urgency": "", "sentiment": "", "reason": f"labeller failed: {type(e).__name__}"}
            save()
            continue
        state["calls"] += len(results)
        state["repaired"] += int(rep)
        state["tokens"]["ver_in"] += sum(r.input_tokens for r in results)
        state["tokens"]["ver_out"] += sum(r.output_tokens for r in results)
        state["verify"][i] = {"urgency": obj.urgency, "sentiment": obj.sentiment, "reason": obj.reason}
        if n % 25 == 0 or n == len(todo):
            save()
            print(f"  verified {n}/{len(todo)} | {usage()}", flush=True)
        time.sleep(args.delay)
    save()

    # ---- 3. keep only rows where both silver labels equal the constructed ones; merge per (plan, seed) ----
    kept, rejected = [], []
    for i, g in sorted(state["gen"].items(), key=lambda kv: int(kv[0])):
        s, v = by_i[int(i)], state["verify"].get(i)
        if v is None:
            continue
        row = {"id": stable_id(g["text"]), "text": g["text"], "category": s.category, "urgency": s.urgency, "sentiment": s.sentiment,
               "language": s.language, "source": "synthetic", "plan": args.plan, "seed": args.seed, "target_class": s.cls,
               "gen_model": model, "silver_model": model, "prompt_version": PROMPT_VERSION, "silver_reason": v["reason"]}
        if (v["urgency"], v["sentiment"]) == (s.urgency, s.sentiment):
            kept.append(row)
        else:
            rejected.append({**row, "silver_urgency": v["urgency"], "silver_sentiment": v["sentiment"]})
    ids: set[str] = set()
    kept = [r for r in kept if not (r["id"] in ids or ids.add(r["id"]))]
    others = _merge(out, kept, args.plan, args.seed)
    _merge(rejected_path, rejected, args.plan, args.seed)

    # ---- 4. report ----
    def cell(s_or_r) -> str:
        return s_or_r.cls if args.plan == "minority" else f"{s_or_r.urgency}/{s_or_r.sentiment}"

    planned = collections.Counter(cell(s) for s in specs)
    gen_c = collections.Counter(cell(by_i[int(i)]) for i in state["gen"])
    kept_c = collections.Counter(r["target_class"] if args.plan == "minority" else f"{r['urgency']}/{r['sentiment']}" for r in kept)
    print(f"\n{args.plan}: {len(kept)} kept, {len(rejected)} rejected; merged into {out} next to {others} rows of other plans; "
          f"{state['calls']} calls, {state['repaired']} repaired; {usage()}")
    print(f"{'cell':16s} planned generated kept dropped")
    for c in planned:
        print(f"{c:16s} {planned[c]:7d} {gen_c[c]:9d} {kept_c[c]:4d} {gen_c[c] - kept_c[c]:7d}")
    print("kept by language:", dict(collections.Counter(r["language"] for r in kept)),
          "| kept urgency:", dict(collections.Counter(r["urgency"] for r in kept)),
          "| kept sentiment:", dict(collections.Counter(r["sentiment"] for r in kept)))
    print("rejections (constructed -> silver):", collections.Counter(f"{r['urgency']}/{r['sentiment']} -> {r['silver_urgency']}/{r['silver_sentiment']}" for r in rejected).most_common(6))
    cats = collections.Counter(r["category"] for r in kept)
    print("kept per category: min", min(cats.values()) if cats else 0, "max", max(cats.values()) if cats else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
