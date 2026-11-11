# Numbers Nobody Publishes: estimating revenue & headcount for a messy licensee list

> Work in progress. Sections below are placeholders that will be filled in as the parts are done.

**Route:** quant: learn the relationship between observable company traits and size from 685k Romanian 2024 filings,
then apply it to the client's licensees as ranges.

## Repository layout

```
scripts/download_data.sh   fetch all raw data (~1.3 GB) into data/raw/ (gitignored)
src/build_dataset.py       filings + trade register → features, exclusion funnel, frozen 80/20 split
reports/                   small generated outputs (funnel, metrics) — committed
notes/decision_log.md      chronological log of choices and dead ends
data/raw, data/processed   gitignored; recreate with the two commands below
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh        # ~1.3 GB, a few minutes
python src/build_dataset.py          # ~2 min
```

## Part 1: The match
_TBD_

## Part 2: The estimate
_TBD_

## Part 3: The call
_TBD_

## Part 4: The room
_TBD_

## Part 5: The register
_TBD_

## Debate topics
_TBD_

## Cut list
_TBD_
