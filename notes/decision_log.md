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

## 2026-11-12: Model ladder (CV on train only, `src/cv_models.py` → `reports/cv_metrics.csv`)

Fresh clone rebuilt with `build_dataset.py`: identical funnel, train 547,680 / holdout 136,971.

### Results (5-fold CV, LightGBM = best rung)
| target | R² | typical miss (×) | within 2× |
|---|---|---|---|
| revenue: global median → industry median → OLS → LightGBM | −0.00 → 0.12 → 0.19 → 0.27 | 3.25 → 2.91 → 3.02 → 2.73 | 33% → 35% → 33% → 37% |
| employees (log1p): same ladder | −0.08 → 0.06 → 0.21 → 0.30 | ~1.5 for all | 81% → 79% → 68% → 72% |

Each rung adds R², but **register-only features cannot tell a big company from a small one.**
LightGBM revenue by true size: <1m RON typical miss 2.3× (overestimates); 1–10m 5.5×; 10–50m 18×; **>50m RON 40× (underestimates)**.
Employees 250+: 27× under. The model shrinks everything toward the median small Romanian firm, and the licensees sit
in exactly the bands where it fails. (Part of this is the usual regression-to-the-mean effect of slicing by the true
value, but the size of it rules out shipping register-only estimates for large firms.)
Note: employee "within 2×" overall is flattering; most firms have 0–3 staff, so log1p errors are small in absolute terms.

### Probe: add headcount as a LinkedIn-style band (1 fold)
Revenue model + employee band (0, 1, 2–10, 11–50, 51–200, 201–500, 501–1k, 1k–5k, 5k–10k, 10k+):
typical miss 2.75× → 2.07× overall; **>50m RON 41× → 2.4×** (within 2×: 4% → 42%); 10–50m 18× → 2.4×.
Headcount band is the brief's own "universal proxy" and observable abroad, which makes this a two-stage recipe:
observe the headcount band (LinkedIn / registry) → model revenue | band + industry + portable traits.
Remaining bias in big bands is −1.1 to −1.2 log points (~3× under), so the top bands still need attention.

### Open decision (user)
- [ ] Pivot to a two-stage recipe (observed headcount band → revenue model), or keep register-only + wide ranges?
  Trade-off: needs a headcount band per licensee (LinkedIn lookup = manual per row, but universal and reproducible
  at scale via data vendors); register-only needs nothing but is ~useless above 10m RON.
- [ ] If two-stage: headcount itself is then *observed* (as a band), not predicted. What does the employee
  deliverable become — the band, or a point inside it?

## 2026-11-13: Ladder chosen, sample of 22, LinkedIn back-test

**Decision (user): evidence ladder.** Per licensee, the best tier available: 1 reported figure → 2 observed headcount band
→ revenue model | band + industry + portable traits → 3 register-only model, wide range → 4 "cannot estimate".
Not averaging tiers: averaging a 40×-biased estimate with a 2.4× one just dilutes the good one; disagreement between tiers is a flag.

**Industry codes are portable.** CAEN Rev.2 = NACE Rev.2 (EU) ⊂ ISIC Rev.4: 2-digit divisions identical to ISIC,
4-digit mostly 1:1 via official correspondence tables. UK SIC 2007 and German WZ 2008 extend NACE; China's GB/T 4754 maps to ISIC.

**Sample: 22 of 100**, at least one per trap group + Romanian firms as back-test (`notes/headcount_lookup.md`, column S).
Lookup was time-boxed (~10 min/company); "not found + where I looked" is a valid result.

### Back-test: does the LinkedIn band agree with the filed truth? (Romanian licensees)
| Licensee | Filed 2024 staff | LinkedIn band | Page | Agrees? |
|---|---|---|---|---|
| Yazaki Romania | 3,623 | 1k–5k | Romanian entity page | ✅ |
| Mercedes-Benz Romania | 298 | 201–500 | Romanian entity page | ✅ |
| Cargill Agricultura | 149 | 10k+ | global page only | ❌ ~70× too high |
| Coca-Cola HBC Romania | 1,481 | 10k+ | global page only | ❌ ≥7× too high |

**LinkedIn is right when a country/entity page exists and wrong by an order of magnitude when only the global
page does.** So tier 2 needs a rule: a global-page band is a *group* figure and cannot be used as the entity's
headcount (only as context / upper bound for the group). Small n (4), but the mechanism is clear.

### Other discoveries during lookup
- **Sources are mostly vendor estimates**: Revelio, ZoomInfo (LinkedIn-profile based), PitchBook, EMIS, Tracxn.
  Reported figures only from 10-Q (Fastenal), Companies House (Olivier's Bakery), annual reports (Orion, Wirecard 2019).
- **Absolute vs FTE**: Fastenal reports both (24,679 people / 21,749 FTE). Romanian I20 = average employees.
  (To verify: whether part-timers are counted pro rata, which would make it closer to FTE.)
- **China publishes entity headcounts** (insured staff in SAMR annual reports) but qcc/gsxt are unreachable from abroad
  without a Chinese phone number → at scale this needs a vendor with China access. Siemens Industry Software (Shanghai):
  only an upper bound (East Asia division headcount ≈ 2.4k, Revelio).
- **Name match ≠ entity match**: Steyn Holding S.A., SPF has no active Luxembourg match; the closest is a struck-off
  S.à r.l. (B31697), different legal form. A same-surname company in South Africa can't be it (SPF is a Luxembourg-only form).
- **Dead licensees**: most German Tischlereien on the list are deleted on northdata (e.g. Holtzum UG, terminated 2018);
  the live one shows no filings since 2020. Part 4 candidate: how much of the licensee base still exists?
- **Brands**: Old Spice has no legal entity; the contracting P&G entity is unknown → ask the client.

## 2026-11-13: Tier 2 model (`src/band_model.py`)

Revenue | headcount band (LinkedIn bands, ordinal) + features, LightGBM, 5-fold CV on train.
- **full vs lean features: 2.07× vs 2.09× typical miss.** The register-only extras (has_web, n_caen_authorised, n_reps,
  any_foreign_rep, has_parent, name_group_word) add almost nothing once the band is known, and they are the least
  portable (has_web = 1% of Romanian firms vs ~100% of multinationals). → lean set proposed.
- Per band (lean): 11–50 → 1.75×, 51–200 → 1.66×, 201–500 → 1.55×, 501–1k → 1.52×, 1k–5k → 1.76×.
  **5k–10k and 10k+: only 11 and 7 Romanian firms in train** (4.6× / 8.3× miss, bias −1.5 / −1.1).
  Romania has almost no companies that size, so the model is outside its training data for the sample's big rows.
- **Ranges (80%, from OOF residuals per band, median-corrected):** held-out coverage 0.77–0.81 for bands up to 501–1k
  (0.87 at n=327); **1k–5k only 0.67** (n=178, undercovers). Width (high/low): 0 → 66×, 1 → 17×, 2–10 → 14×,
  11–50 → 9×, 51–200 → 7.5×, 201–500 → 5×, 501–1k → 5.6×, 1k–5k → 6.7×.
- **Tier 3 (register only) ranges, by predicted-revenue decile:** width 47× (D1) to 190× (D10). Coverage is honest
  (0.80) but the ranges are too wide to price a contract on. Confirms tier 3 = "order of magnitude at best".

## 2026-11-14: Decisions on the tier 2 model (user)

### 1. Lean feature set
**Chosen:** headcount band + industry (NACE 2- and 4-digit) + age + legal form (SA) + capital city + foreign parent.
**Why:** CV accuracy is practically identical (typical miss 2.09× lean vs 2.07× full). The six dropped features are
coded from the Romanian register and mean something different abroad: `has_web` is 1% of Romanian filers but ~100% of
multinationals, so the model would treat every foreign licensee as an outlier. When two models score the same, take the
one whose inputs you can actually observe for a company in Mexico or Nigeria. Bonus: 6 inputs per licensee
instead of 12, so the recipe is cheaper to run at scale.
**What it costs:** nothing measurable in CV; possibly some signal for Romanian firms specifically.

### 2. 80% ranges
**Chosen:** each estimate carries the central 80% of out-of-fold residuals for its headcount band (P10–P90),
re-centred on the band's median residual.
**Why:** a range has to be narrow enough to decide a pricing tier and honest enough to bet on. 90% would be ~1.5× wider
(e.g. 201–500 band: 5× → ~8× low-to-high), which for most bands spans two pricing tiers and decides nothing.
80% means **1 in 5 true values will fall outside the range, by design**; that is stated, not hidden.
**How anyone can check it a year later:** count how many reported figures land inside their ranges. Should be ~80%.
Held-out coverage in CV was 77–81% for bands up to 1k; **1k–5k was only 67%** → those ranges are too narrow, flag it.
**Debate-topic link:** this is the answer to "a range you would bet on": widths from out-of-sample errors, honesty
measured by coverage.

### 3. Productivity ratio for non-Romanian licensees
**Chosen:** scale the model's revenue (learned on Romanian firms) by the ratio of output per worker,
country / Romania, at **market exchange rates** (World Bank: GDP in current US$ ÷ employed persons).
**Why:** a 500-person firm in Germany or the US sells far more than a 500-person Romanian firm. The model only knows
Romanian revenue per head, so without a correction every foreign estimate is biased low by roughly the productivity gap.
**Why nominal, not PPP:** the client prices contracts in money at market rates; PPP would answer "how much do they
produce in real terms", not "how much revenue do they book".
**Why GDP per worker:** free, reproducible, available for every country on the list, one number per country.
**Known weaknesses (put in README):** it is an economy-wide average, so sector mix leaks in (a US software firm vs the
US average); it assumes the Romanian *shape* of revenue-by-size holds abroad, shifted by a constant; and the ratio's own
uncertainty is **not** inside the 80% range, so foreign ranges are, if anything, too narrow.
**Alternatives considered:** Eurostat SBS turnover per person by sector (closer to firm revenue, but EU only);
apply the model only to countries similar to Romania (honest, but leaves most of the sample without a tier 2).

### Productivity ratios: first results (`src/productivity.py` → `reports/productivity_ratios.csv`)
World Bank 2024, GDP current US$ per employed person, vs Romania (US$ 48.9k): US 3.58, CH 4.03, DE 2.26, GB 2.24,
FR 2.19, SE 2.38, FI 2.32, MX 0.63, CN 0.52, BR 0.44, **IN 0.13, PK 0.10, NG 0.05**, **LU 5.78, IE 4.55**.
RON/EUR 2024 average 4.9746 (ECB).
**Concern at both ends.** Economy-wide averages include informal work and subsistence farming (NG, IN, PK) and
tax-driven GDP / cross-border commuters not counted as resident employment (IE, LU). A formal-sector multinational
subsidiary in Lagos is far more productive than the Nigerian average, so NG ×0.05 would cut Unilever Nigeria's estimate 20×.
**Check before trusting:** the sample has listed firms with reported revenue in those countries (Unilever Nigeria,
Siemens Ltd India, Fastenal US, Orion FI): compare model × ratio against reported revenue, then decide on capping.

### 4. Above 5k employees: reported revenue is the answer, the model is a cross-check (user)
**Why:** the training data has 11 firms at 5k–10k and 7 at 10k+; the model's typical miss there is 4.6× / 8.3×.
Firms that size are almost always listed or publicly reported, so the cheap reported figure beats the model.
**Consequence:** the ladder runs per number, not per company. Headcount and revenue can sit on different tiers.
Reported revenues for listed firms in NG / IN / US / FI double as the test of the productivity ratio.

### 5. ANAF (mfinante.gov.ro) for Romanian firms; LinkedIn kept for the back-test only (user + discussion)
ANAF shows the newest balance sheet (FY2025 judging by the inactivity declarations "până la 31.12.2025"), so it is the
answer for Romanian rows: Cargill 134, Coca-Cola HBC 1,515, DRUID S.A. 122, FINTECH OS SRL 148.
**The LinkedIn band stays in the sheet** although it's wrong, because it answers a different question: *would the cheap
method have found it without a registry?* For Cargill it would have said 10k+ vs 134 (≈70× too big): the multinational
subsidiary sized as the whole group, i.e. the client's exact fear. In Romania ANAF catches it; for a subsidiary in
Brazil or Singapore nothing would. Not negligible: it is the largest single error source in tier 2, and it is
invisible exactly where tier 2 is needed.

### More ANAF findings
- **ALIN PANIFICAȚIE SRL and PANIFICAȚIE STRAJA SRL both filed sworn declarations of no activity since founding,
  through 31.12.2025.** Registered, "alive" by status code, never operated. Shells by their own declaration.
  Together with the dead Tischlereien, Part 4 material: a visible share of the small licensees are not operating companies.
- DRUID and FintechOS: absent from the 2024 open-data filings, yet ANAF has a balance sheet. The open-data dump is
  incomplete (late filers? a third format?). Matters for Part 5: register coverage ≠ filing coverage.

## 2026-11-15: Ladder applied to the sample (`src/ladder.py`): the model fails on big companies

User calls before the run: Uline Inc. counted as SA (not 100% clear: US private corporation); Mistral SAS as non-SA
(ambiguous form, growth-stage AI group); EQT modelled but flagged as a weak fit (fee income on client capital, not
turnover). Data fixes: Fastenal's 7,051.9 was **9 months of 2026**, not FY2025 → FY2025 10-K: USD 8,200.5m
(TTM to Sep 2026: 9,079.3m). Siemens Ltd India: PitchBook "2,037,975" had no unit → screener.in TTM Rs 18,897 cr
(INR 189.0bn); 2025 sales −23% YoY, consistent with the energy demerger.

### Back-test, Romanian licensees (model vs filed revenue; RON, no ratio)
| | band | model | truth | off | in 80% range? |
|---|---|---|---|---|---|
| Yazaki | 1k–5k | 1,329m | 1,095m | 1.2× | ✅ |
| Coca-Cola HBC | 1k–5k (true) | 1,311m | 4,002m | 3.1× | ❌ |
| Mercedes-Benz RO | 201–500 | 298m | 2,917m | 9.8× | ❌ |
| Cargill Agricultura | 51–200 (true) | 251m | 3,514m | 14× | ❌ |
| Cargill Agricultura | 10k+ (LinkedIn, wrong) | 2,898m | 3,514m | 1.2× | ✅ right for the wrong reason |
**1 of 4 in range with the true band.** Trading and distribution subsidiaries of multinationals (Cargill: grain
trading; Mercedes-Benz RO: car import and distribution) book the group's sales through a small local team.
Revenue per head is 10× the Romanian norm for their band, and industry + foreign-parent features don't capture it.

### Ratio check, foreign rows with reported revenue (EUR m)
| | country | reported | model × ratio | reported / model |
|---|---|---|---|---|
| Orion | FI ×2.32 | 1,890 | 376 | 5.0× |
| Uline (Forbes est.) | US ×3.58 | 7,168 | 1,142 | 6.3× |
| Fastenal | US ×3.58 | 7,257 | 809 | 9.0× |
| Siemens Ltd | IN ×0.13 | 1,918 | 35 | **55×** (54× worse than with no ratio: 7.2×) |
| TfL | GB ×2.24 | 10,830 | 183 | 59× (public body, grants) |
The ratio points the right way for rich countries (US: 32× → 9×; FI: 12× → 5×) but is far too small, and for
India it makes things much worse (the economy-wide average is not a listed multinational).
**Every licensee with known revenue is underestimated by the model, by 5–60×.** The 80% ranges are calibrated on the
Romanian population (mostly micro firms) and are not honest for this client's population (large, productive,
multinational). This is selection: the licensees are not a random draw from the training data.

### 6. Withhold model revenue for big and multinational firms (user, on recommendation)
**Rule:** tier 2 revenue is shown only for firms under 1,000 staff, without a foreign parent, in a country whose
productivity ratio is credible. Otherwise the file says "Withheld", gives the reason, and still shows the model's output
in the cross-check column, so the decision is visible rather than hidden.
**Why:** every sample firm with known revenue that has 1k+ staff or a foreign parent was underestimated 3–60×. "A wrong
number costs more than a missing one": a withheld cell is visible; a plausible €59m for Siemens Mexico is not.
**Result:** of 23 rows, 10 have tier 1 revenue, 1 tier 2 (Olivier's Bakery), 8 withheld, 4 tier 4. Headcount is
available for 18 of 23 (exact or band), which is where the method is strongest.
**Alternatives rejected:** (b) correct the model by the observed 6–10× gap: estimated on 9 rows, tested on the same rows;
(c) keep numbers as low-confidence: someone would still price on them.
**Next direction (user idea, for the README):** revenue ≈ headcount × wage ÷ industry staff-cost share. Staff-cost
share differs hugely by industry (trading 1–3%, manufacturing ~15%, services 40–60%), which is the Cargill / Mercedes
failure. Still country-average wages, though, so it would not fix the multinational premium on its own. Validate on
registries that publish both revenue and staff costs (Romania, UK) before trusting it.

## 2026-11-15: Holdout evaluation (`src/evaluate.py`, run once) and Part 5

**Holdout (137k firms never seen):** tier 2 typical miss 2.09×, within 2×: 48%, **80% ranges held 80.0%** of the time,
per band 78–81% up to 1k (1k–5k: 92%, n=37). **5k–10k: 1 of 4 in range, ~4× too low** (n=4).
Tier 3 (register-only): 2.73× overall, 7.7× at 11–50 staff, 117× at 1k–5k.
→ The model is honest for the average Romanian company and breaks exactly at the licensees' size. The ranges are
calibrated for the population they were learned on, not for a client list of multinationals.

**Part 5 (`src/part5_register.py`):** register 3,924,395 unique CUIs → alive + company form 1,294,677 (denominator)
→ filed 2024 accounts 808,968 → <1k staff 808,732 → no foreign parent 808,551 → not finance 797,783.
**797,783 companies, 61.6% of the filtered register.** Top industries: NACE 47 retail (113,792), 49 land transport
(63,337), 46 wholesale (56,060). The binding constraint is the headcount: the register itself has neither headcount
nor main activity, only the filings do. Weak line: TARA_FIRMA_MAMA is filled for <1% of firms, so the foreign-parent
filter removes only 181 companies and most multinational subsidiaries slip through. 21,349 inside the line had zero
2024 turnover (dormant), and the model would still output a number for them.

## TODO / open questions
- [x] Baseline: industry-median model. Then OLS, then LightGBM. Compare on CV inside train only. (2026-11-12)
- [ ] Prediction intervals from out-of-fold residual quantiles, possibly per size/industry bucket.
- [ ] Choose the sample (~20–25) from the licensee list covering every trap category.
- [ ] Back-test on the Romanian licensees whose true figures we have.
