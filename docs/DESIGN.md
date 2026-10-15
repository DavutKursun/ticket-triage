# Design notes — ticket-triage

## Why two paths
Clients ask "fine-tune or API?". The honest answer depends on volume, latency, privacy and how
often labels change. This project measures all four on the same data so the recommendation is a
table, not an opinion.

## Labels
- Category: gold from Bitext (11 categories; 27 intents collapsed).
- Urgency / sentiment: silver labels from an LLM (rubric in `ticket_triage/rubric.py`,
  `PROMPT_VERSION` recorded on every row). Quality measured on 200 hand-checked rows (Cohen's kappa).
  Known limitation: silver labels inherit the LLM's biases; the hard set is hand-labelled to
  counter that.
- Hard set: 300 realistic messages, hand-edited and hand-labelled. The number clients should trust.

## Structured output
Pydantic `Triage` with `Literal` enums; values normalised case-insensitively. Groq: JSON mode +
Pydantic + one repair retry (json_schema mode corrupted UTF-8 in the invoice project, so it is
opt-in). Claude: `messages.parse(output_format=Triage)`.

## Cost accounting
Measured input/output tokens per call × published list prices (`costs.py`, checked 2026-10-09).
The default engine runs on Groq's free tier; the README quotes the cost "as if" on Claude Haiku
so the comparison is meaningful for a paying client. Batch (50% off) and prompt-cache variants
are computed from the same token counts.

## Fine-tuned path (M3)
xlm-roberta-base, three classification heads, loss = sum of cross-entropies, 3 epochs, early
stopping on dev macro-F1, Kaggle T4. Export: ONNX + int8 dynamic quantisation via optimum;
latency measured batch 1 and 32 on CPU.

## Findings
_Milestone 4._

## Next steps
Reply drafting, Zendesk/Freshdesk routing, active learning on low-confidence tickets.
