# Reviewing hard_draft.csv → hard.csv

1. `hard_draft.csv` holds 300 candidates drafted by claude-haiku-5-5. Nothing is final: read every row, edit it, and save the result as `data/hard/hard.csv` with the same header; leave `hard_draft.csv` untouched so the diff shows your changes.
2. Edit text freely so each row reads like a ticket you would actually receive. Keep 2–6 sentences; keep the two-issue flavour (90 rows have a minor second issue) and the typos (75 rows) where they feel natural; vary the openings.
3. Fix labels: `category` is the issue that needs action first; `urgency` follows the rubric in `ticket_triage/rubric.py` by consequence (urgent = money being lost / service down / deadline today, high = blocked, medium = inconvenience, low = question), never by tone; `sentiment` is tone only, so a thankful customer with a duplicate charge is positive + urgent.
4. Delete weak rows (templated, implausible, near-duplicate, wrong language) by removing the line. Keep at least 40 urgent, 30 positive and 60 Turkish overall; the draft has 48 / 36 / 60, so there is slack.
5. Keep `id` exactly as it is even after rewriting the text. It is the join key for results and for the diff; never regenerate it.
6. Keep `language` (`en`/`tr`) and `split` (`dev`/`test`, 150 each, balanced by language and category). When you delete rows, take them from whichever split is larger so the halves stay even.
7. Turkish rows: fix anything that reads like translated English. 13 rows are typed without ş/ğ/ı/ö/ü/ç (6 of them on purpose as typos); that is realistic phone typing, so keep or restore as you see fit. Label values stay English.
8. Check first: `03b8cf23717b` (feedback). The generator wrote it as low urgency although the plan asked for high; relabel it or sharpen the text.
9. The draft labels are the generation targets, not the model's judgement of its own text, so expect a few more mismatches like that one; when text and label disagree, change whichever is wrong.
10. Validate when done: `python -c "from ticket_triage.data import load_hard as l; d=l(); print(len(d), d.language.value_counts().to_dict(), d.split.value_counts().to_dict(), d.urgency.value_counts().to_dict(), d.sentiment.value_counts().to_dict())"` and expect ≥ 280 rows, 60 tr, two near-equal splits, ≥ 40 urgent, ≥ 30 positive.
