# Numbers Nobody Publishes: estimating revenue & headcount for a messy licensee list

**Client file:** [`deliverable/licensee_size_estimates.xlsx`](deliverable/licensee_size_estimates.xlsx) (sheet 1: 23 sampled
licensees; sheet 2: how to read it; sheet 3: the 77 not assessed). Every figure carries its tier, source, scope (entity or
group), date and confidence.

**In one paragraph.** I learned the relationship between company traits and revenue from 685,000 Romanian 2024 filings,
then built an *evidence ladder* that gives each number the best evidence available: **1** a reported figure, **2** an
observed headcount band turned into revenue by the model, **4** no estimate, with the reason and what would fix it.
Headcount and revenue climb the ladder separately. On a 23-row sample chosen to hit every trap in the list, the method
produces a defensible headcount for 18 rows and a defensible revenue for 11. The model is honest on the average Romanian
company (80% ranges held 80.0% of the time on 137k unseen firms), **but it underestimates every large or multinational
licensee with known revenue by 3–60×, so I withheld its output there.** Saying where it breaks is the main result.

## Repository layout

```
notes/brief.md                 the assignment
notes/decision_log.md          every choice and dead end, in order (this README is distilled from it)
notes/headcount_lookup.md      the manual lookup sheet: 100 licensees sorted by trap type, sources tried
notes/sample_evidence.csv      what was found for the 23 sampled rows (clean units, quality labels)
notes/sample_inputs.csv        model inputs for the non-Romanian rows
src/build_dataset.py           filings + trade register → features, exclusion funnel, frozen 80/20 split
src/cv_models.py               model ladder by 5-fold CV: global median → industry median → OLS → LightGBM
src/band_model.py              tier 2: revenue | headcount band (+ calibrated 80% ranges)
src/productivity.py            World Bank output per worker vs Romania; ECB exchange rates
src/ladder.py                  applies the ladder to the sample → client file, back-test, ratio check
src/evaluate.py                one-time evaluation on the frozen holdout
src/part5_register.py          Part 5 count
reports/                       every number quoted below, as small CSVs
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
bash scripts/download_data.sh                                     # ~1.3 GB raw data
python src/build_dataset.py && python src/cv_models.py && python src/band_model.py
python src/productivity.py && python src/ladder.py && python src/evaluate.py && python src/part5_register.py
```

## Part 1: The match

**How I chose the sample.** I sorted all 100 names into 11 trap types (group subsidiaries, listed groups, US private
firms, financial firms, public bodies, startups, brands, ambiguous names, defunct firms, holdings, tiny firms) and took
at least one from each. I added the Romanian licensees on top, because there I have the true figures and can back-test
the cheap methods. 23 rows in total. Lookups were time-boxed to about 10 minutes per company; "not found, and where I
looked" counts as a result.

**What a match needs.** A registry identifier wherever one exists: Romanian CUI, UK company number, Chinese unified
credit code (via QCC), Luxembourg RCS. A name alone was never treated as a match.

**What the matching taught me** (each one would quietly corrupt an automated pipeline):
- **Same name, different company.** FintechOS's real entity (FINTECH OS SRL) is missing from the 2024 open-data filings,
  while *FINTECHOS EASY SOLUTIONS SRL* (8 staff, a different firm) is there. A fuzzy name match would pick the decoy.
- **Same name, wrong legal form.** The only "Steyn Holding" in Luxembourg is a struck-off *S.à r.l.*; the licensee is an
  *S.A., SPF*. Not a match. (Same-surname companies in South Africa can't be it: SPF is a Luxembourg-only form.)
- **Brands aren't entities.** Old Spice is a P&G brand: the question to the client is *which P&G entity signed*.
- **Ambiguous names need a stated choice.** "Orion Corporation", no country: I chose Orion Oyj (Finnish pharma) over
  Orion Corp (Korean confectionery) as a best guess, and the file says so.
- **One company, many registrations.** Banca Transilvania has ~70 branch registrations in the Romanian register; Coca-Cola
  HBC Romania has three after moving between counties.
- **Registered isn't operating.** Two Romanian bakeries are "alive" in the register but filed sworn declarations that
  they have *never operated* since founding (through 31.12.2025). One German joinery was dissolved in 2018.

## Part 2: The estimate

**Route: quant.** Learn revenue from observable traits on the Romanian filings, apply to licensees.

**Training data.** The brief's file (`web_uu`, abbreviated balance sheets) contains only small companies: the 99th
percentile is 18 employees, and Yazaki, Cargill and Coca-Cola HBC are absent. I combined it with the full-format file
(`web_bl_bs_sl`, up to 30,900 employees). 899,961 filings → 684,651 modelling rows after removing non-company forms,
dormant shells (zero turnover), and pure holdings (their "turnover" is dividends). The 80/20 split was frozen before any
modelling. Full funnel: `reports/funnel.csv`.

**Only portable features.** Everything must be observable for a company in Mexico or Nigeria: industry, age, legal form,
capital-city location, foreign parent. The filing type was excluded: it is decided by size (leakage) and doesn't exist abroad.
Industry codes travel: CAEN = NACE (EU) ⊂ ISIC (UN), identical at 2 digits; UK SIC and German WZ extend NACE.

**Step 1: register traits alone can't size a company.** LightGBM beat the industry median and OLS (R² 0.27), but it
missed companies above 50m RON by **40×**: it pulls everyone toward the typical small Romanian firm (`reports/cv_metrics.csv`).

**Step 2: add an observed headcount band.** With LinkedIn-style bands as an input, the typical miss falls to 2.1× overall
and 1.5–1.8× for 11–1,000 staff (`reports/band_model_cv.csv`). The *lean* feature set (band, industry, age, legal form,
capital, foreign parent) scored the same as the full one (2.09× vs 2.07×) and drops the Romanian-only register fields, so
I kept lean. Ranges: the central 80% of the model's out-of-fold errors in each band. On the frozen holdout they held
80.0% of the time (`reports/holdout_eval.csv`).

**Step 3: cross a border.** A 500-person German firm sells more than a 500-person Romanian one, so foreign estimates are
scaled by output per worker relative to Romania (World Bank 2024, market exchange rates, not PPP, because the client
prices in money): US 3.6×, DE 2.3×, GB 2.2×, MX 0.6×, IN 0.13×, NG 0.05× (`reports/productivity_ratios.csv`).

**Headcount itself** comes from the best public source per country: registry accounts (UK Companies House, Romanian
ANAF), annual reports and 10-Qs, then vendors (Revelio, ZoomInfo, PitchBook, EMIS, Tracxn), then LinkedIn bands. Each
is labelled *reported*, *vendor estimate* or *inferred*, and *entity* or *group*.

## Part 3: The call

**Would I ship it? Partly.** I would ship the headcounts, and revenue where it is reported. I would not ship model
revenue for this client's large and multinational licensees, and the file withholds it.

The evidence, from the only rows where I know the truth:

| Check | Result |
|---|---|
| Holdout, 137k Romanian firms | typical miss 2.1×, 80% ranges hold 80.0% ✅ |
| Holdout, the 4 firms with 5k–10k staff | 1 of 4 in range, ~4× too low ❌ |
| Back-test, Romanian licensees, true headcount | Yazaki 1.2× ✅ · Coca-Cola HBC 3× ❌ · Mercedes-Benz RO 10× ❌ · Cargill 14× ❌ |
| Foreign licensees with reported revenue, model × ratio | Orion 5× · Uline 6× · Fastenal 9× · Siemens India 55× · TfL 59×, all too low ❌ |

The model is calibrated on the population it learned from, and this client's list is not that population. Trading and
distribution arms of multinationals (Cargill grain trading, Mercedes-Benz car import) book the group's sales through
small local teams. The productivity ratio points the right way for rich countries but is far too small, and for India
it makes things worse: an economy-wide average isn't a listed multinational. So eight model revenues are marked
**Withheld**, with the model's number still shown so the decision is visible. A missing number is visible; a plausible
€59m for Siemens Mexico (6,500 staff) is not.

**What my checks can't see.** The back-test is 4 companies and the foreign check 5: evidence of a mechanism, not a
statistic. Vendor headcounts (most of the sample) are unverified against registries. LinkedIn's global pages overstate
subsidiaries by up to 70× (Cargill RO: 134 filed vs 10k+ on LinkedIn), and that error is invisible wherever there is no
registry to catch it.

**Next directions.** (1) Revenue ≈ headcount × local wage ÷ the industry's staff-cost share of revenue. The share
ranges from ~1–3% in commodity trading to 40–60% in services, which is exactly the Cargill/Mercedes failure.
Validate it on registries that publish both (Romania, UK) before trusting it. (2) Train on registries that contain large
firms (UK Companies House full accounts). (3) China's regulator publishes insured-staff counts per entity, reachable only
through a vendor with access from inside China.

## Part 4: The room

**The licence list is frozen in time; the companies on it are not.** At the large end, at least six names no longer exist
as the entity that signed: VMware (now Broadcom), ANSYS (now Synopsys), Credit Suisse (merged into UBS), Siemens Gamesa
(absorbed into Siemens Energy), WK Kellogg (acquired by Ferrero), Wirecard (insolvent since 2020). In each acquisition, the
licence now sits inside a far larger group. That is the client's "small tier, multinational group" fear, created *after*
signing by M&A, and it is the cheapest to find: a list of change-of-control events, not an estimate. At the small end, the
licensees often aren't operating at all: of the tiny firms I checked, two Romanian bakeries declared they had never operated,
and the German joineries were either deleted from the register or silent since 2020. **A routine check of registry status
and ownership against the licence list would find renegotiation leverage at the top and dead accounts at the bottom, and
neither needs a single revenue estimate.** *(The acquisitions are public knowledge and were not part of the timed lookup;
verify the closing dates before sending.)*

## Part 5: The register

**Alive** = status 1048 (*în funcțiune*) with no dissolution, liquidation, insolvency, deregistration or
temporary-inactivity code; commercial company forms only. **The line:** the model needs a headcount band and a main
activity code. The register has neither, only the filings do. So: filed 2024 accounts, under 1,000 staff, no foreign
parent, not finance/insurance (where revenue isn't turnover).

| | |
|---|---|
| Register, unique companies | 3,924,395 |
| Alive + company form (denominator) | 1,294,677 |
| **Inside the line** | **797,783 (61.6%)** |
| Largest industries | Retail (NACE 47) 113,792 · land transport (49) 63,337 · wholesale (46) 56,060 |

The weak part of the line: the register records a parent company for under 1% of firms, so the "no foreign parent" filter
removes only 181 companies, and most multinational subsidiaries pass through it. 21,349 companies inside the line had zero
turnover in 2024, and the model would still output a number for them. Code: `src/part5_register.py`.

## Debate topics

*(Draft from my notes; edit into your own words.)*

**Where I wouldn't trust an AI tool.** With facts that look like data: a figure, a date, a unit. In this project, a
headcount pasted with the wrong source link, a revenue figure that was 9 months rather than a year (Fastenal), and a
number with no unit (PitchBook) all looked fine until checked against the original filing. My rule: every number in the
client file links to a primary source I opened, and anything an AI tool wrote from memory is marked "verify".

**Knowing the dataset is wrong at scale, with no complete answer key.** Check it against the slices where the truth
*is* known (a registry that publishes both numbers), and check internal consistency: revenue per head by industry, ranges
that hold the share they claim. The Romanian back-test is a small version of this: it found the LinkedIn group-page error
and the trading-subsidiary error without any external answer key.

**Ranges you'd bet on.** Widths come from the model's own out-of-sample errors, per size band, not from intuition. A year
later, honesty is measurable: of the figures that became public, about 80% should sit inside their 80% ranges. If far
fewer do (my large-licensee check: 0 of 5), the ranges were not honest for that population, and the fix is to say so and
withhold, not to keep publishing them.

## Cut list

- **Time went to:** the training data (finding that the brief's file lacks large firms), the model ladder, and the manual
  lookups. Each lookup took longer than the modelling it fed.
- **Sampled 23 of 100** on purpose: one or more per trap type beats 100 shallow rows.
- **Not done:** the wage × staff-cost-share model (idea logged, not validated in time); Unilever Nigeria's reported revenue
  (it would have tested the Nigeria ratio); verifying every model input for foreign rows (founding years partly
  missing); China's registry headcounts (blocked from abroad); paid vendor reports (EMIS, ~€15/company, worth a line
  as the at-scale alternative); automating the name matching; FintechOS and DRUID were looked up but kept out of the sample.
