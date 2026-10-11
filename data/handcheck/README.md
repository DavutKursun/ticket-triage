# Hand-check (200 messages)

`python scripts/agreement.py --make-templates` writes `dev_100.csv` and `test_100.csv`
(stratified by silver urgency). Davut fills `urgency`, `sentiment` (and optionally `note`)
WITHOUT looking at the silver columns first (cover them in the spreadsheet), then
`python scripts/agreement.py` reports accuracy and Cohen's kappa → README.
