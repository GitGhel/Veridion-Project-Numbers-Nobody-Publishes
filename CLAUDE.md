# Project context for Claude Code

Veridion hiring challenge "Numbers Nobody Publishes" (Economic Data Scientist). **Deadline: 2026-11-16.**
Estimate annual revenue + employee count for a messy 100-row licensee list (`data/raw/licensee_list.csv`),
deliver a client file + README + code in this repo. The full brief is in `notes/brief.md`.

## Who you're working with
Finance background (ex-BCR controlling, ASE FABBV, DOFIN master's), reads Romanian. Comfortable with pandas/numpy/
matplotlib and econometric forecasting; explain ML/data-engineering choices, skip accounting basics.
The graders want the *user's* reasoning: do the wrangling and code, but leave judgment calls (sample choice,
which variables/proxies to trust, "would you ship it") to the user, with enough explanation to defend them.

## State of play
- Route chosen: **quant** (learn size from Romanian 2024 filings, apply to licensees as ranges).
- `src/build_dataset.py` builds `data/processed/{companies,train,holdout}.parquet`; funnel in `reports/funnel.csv`.
- **Holdout is frozen**: do not read `holdout.parquet` except in the final evaluation step.
- All decisions and discoveries so far: `notes/decision_log.md`. Keep appending to it; the README is distilled from it.

## Key decisions (see decision log for why)
- Train on `web_uu` (abbreviated) + `web_bl_bs_sl` (full) filings; the brief's web_uu alone has only small firms.
- Filing type is NOT a feature (it's a function of size = leakage, and doesn't exist abroad).
- Only "portable" features (observable for a company in any country). County reduced to capital-city flag.
- Blank employee count = 0. Holdings (CAEN 6420) and non-company legal forms excluded.
- **Evidence ladder** per licensee: 1 reported → 2 headcount band → revenue model → 3 register-only model → 4 cannot estimate.
- Tier 2 = **lean** LightGBM (band, NACE2/4, age, SA, capital, foreign parent), **80%** ranges per headcount band.
- Foreign revenue scaled by **output-per-worker ratio vs Romania** (World Bank, nominal). Sample = 22 rows (S column).

## Next steps
1. ~~CV ladder~~ (`cv_models.py`), ~~tier 2 model + ranges~~ (`band_model.py`), ~~sample + headcounts~~ (`notes/headcount_lookup.md`).
2. Productivity ratios (`productivity.py`), then apply the ladder to the 22 sample rows → client file.
3. Back-test on Romanian rows; final holdout evaluation (only now may `holdout.parquet` be read).
4. Part 5: count register companies the model can score (alive filter defined in the decision log).
5. README from the decision log; Part 4 insight (candidate: share of dead licensees).

## Conventions
- Raw data is gitignored; `scripts/download_data.sh` recreates it. Small outputs go in `reports/`.
- Never store personal data from `od_reprezentanti_legali.csv`; reduce it to counts.
