# Decision log

Working notes, in the order things happened. The README is distilled from this.

## 2026-11-11: Data and training set

**Route: quant.** Background in controlling / econometrics (Markowitz, inflation forecasting).
The training files also serve Part 5, so the time spent on them pays off twice.

### Discoveries
1. **The training file the brief points to only contains small companies.** `web_uu_an2024` = "bilanț format prescurtat",
   the abbreviated balance sheet, which only companies under the size thresholds may file. 822k firms, 99th pct = 18 employees, max = 737.
   The large Romanian licensees (Yazaki, Coca-Cola HBC, Cargill, Mercedes-Benz RO) are **absent** from it.
   They are in `web_bl_bs_sl_an2024` (full balance sheet, 78k firms, max 30,900 employees).
   A model trained on web_uu alone would cap every licensee at small-firm size.
   → Train on both files combined.
2. **Filing type is not a feature.** Which balance-sheet format a firm files is decided by its size, the very thing we
   predict (leakage), and it does not exist outside Romania. Kept only as a diagnostic column.
3. **On data.gov.ro, the `.csv` files are column legends and the `.txt` files are the data.** I13 = net turnover, I20 = avg employees.
4. **A blank employee count means 0.** Median revenue: blank 43k RON, explicit 0 → 51k, 1 employee → 181k. Recoded blanks to 0
   instead of dropping 145k rows (that would have removed most one-person companies and biased the model upwards).
5. **Date format changed in Sept 2024.** New registrations carry a timestamp ("01/10/2024 11:11:17") and new-style
   registration codes (J2024…). The naive parser silently lost 14k rows before this was fixed.
6. **A CUI can have several register rows** (company moved county → re-registered, e.g. Coca-Cola HBC has 3).
   Keep the alive registration, age from the earliest one.
7. **Revenue and headcount decouple by business model.** Cargill Agricultura: 3.5bn RON, 149 staff (trading);
   Yazaki Romania: 1.1bn RON, 3,623 staff (manufacturing). Model them separately.

### "Alive" definition (used for Part 5 and as a flag)
Status code 1048 (în funcțiune) and none of: deregistered (1084), dissolution (1049, 1076, 1094, 1098, 1100, 1109, 1111, 1113, 1120),
liquidation (1052), bankruptcy (1070, 1139), insolvency (1073, 1083, 1105, 1107, 1121), temporary inactivity (1074).
Training does NOT require alive: a firm that filed for 2024 and was dissolved in 2025 still shows the 2024 relationship.

### Exclusion funnel → `reports/funnel.csv`
899,961 filings → 684,651 modelling rows. The big cut is turnover > 0 (−210k dormant shells).
Excluded: non-company legal forms (PFA/II/IF etc.), pure holdings (CAEN 6420: their "turnover" is dividends).
Split 80/20, seed 42, before any modelling. Holdout = 136,971 rows, untouched.

### Feature design principle: only "portable" variables
Everything must be observable for a German carpenter or a US private firm too: industry, age, legal form (SA ≈ AG/plc/Inc),
capital-city location, foreign parent, website, number of activity codes, number of directors, foreign director, group words in the name.
County is NOT portable → reduced to "capital city".

### First look at variable coverage (train split)
Register-side signals are **sparse**: website 1%, parent company <1%, legal-representatives rows only 2.2% of firms
(the open file covers a subset). Correlation with log revenue: age 0.24, n_caen_authorised 0.17, SA 0.11, n_reps 0.09.
Industry not yet tested (categorical); expected to be the main signal.

## TODO / open questions
- [ ] Baseline: industry-median model. Then OLS, then LightGBM. Compare on CV inside train only.
- [ ] Prediction intervals from out-of-fold residual quantiles, possibly per size/industry bucket.
- [ ] Choose the sample (~20–25) from the licensee list covering every trap category.
- [ ] Back-test on the Romanian licensees whose true figures we have.
