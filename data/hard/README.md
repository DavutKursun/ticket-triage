# Hard set (300 messages) — the number clients should trust most

`hard.csv` columns: `id,text,category,urgency,sentiment,language,split`

- 300 realistic tickets: 2–6 sentences, two issues in one message, typos, mixed case, pasted
  order numbers, 60 in Turkish (`language=tr`), the rest `en`.
- Every row is **hand-labelled by Davut** after drafting (scripts/draft_hard.py produces
  `hard_draft.csv` with the LLM's suggestion; Davut edits text and labels, then saves as `hard.csv`).
- `split` is `dev` or `test`, 150 each, balanced by category as far as possible.
- Labels use the rubric in `ticket_triage/rubric.py`. Two-issue messages take the category of the
  issue that needs action first.
