"""Apply the evidence ladder to the sample and write the client file.

Each number (headcount, revenue) climbs the ladder on its own:
  tier 1  reported figure (filing, annual report, 10-K/10-Q, registry accounts)
  tier 2  observed headcount band -> lean LightGBM revenue model (Romanian 2024 filings),
          scaled by the country's output-per-worker ratio vs Romania, 80% range from OOF residuals
  tier 4  cannot estimate (brand, unmatched entity, defunct company): say why and what would resolve it
(Tier 3, register-only, is never used for the sample: its ranges span 50-190x, and its inputs are Romanian-register
 fields that do not exist abroad. See decision log.)

Above 5k employees the reported revenue is the answer and the model is only a cross-check (decision log #4).

Inputs
  notes/sample_evidence.csv  what was found per licensee (headcount, revenue, sources, quality)
  notes/sample_inputs.csv    model inputs for non-Romanian rows (Romanian rows: from the register by CUI)
  data/processed/{train,companies}.parquet, oof_band_model.parquet, reports/{productivity_ratios,fx}.csv
Outputs
  deliverable/licensee_size_estimates.xlsx  the client file (+ .csv of the main sheet)
  reports/backtest_romania.csv              model vs filed truth, with true band and with LinkedIn band
  reports/ratio_check.csv                   model x productivity ratio vs reported revenue, foreign rows
Run: python src/ladder.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from band_model import BAND_LABELS, FEATURES, Q_HI, Q_LO, fit, make_X  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, REPORTS, NOTES, OUT = ROOT / "data/processed", ROOT / "reports", ROOT / "notes", ROOT / "deliverable"
COLS = FEATURES["lean"]
BIG_BANDS = {"5k-10k", "10k+"}            # Romania has 18 firms this size: outside the training data
POOL_FROM = "1k-5k"                       # thin bands borrow their range from all bands >= 1k
RO_CUI = {"IN-016": 15412557, "IN-020": 8915840, "IN-022": 18927698, "IN-024": 474152, "IN-030": 5022670}
NOT_MODELLED = {"IN-030": "bank: not in the turnover filings the model learned from"}
RATIO_UNRELIABLE = lambda r: r < 0.2 or r > 4  # NG/IN/PK (informal economy) and LU/IE (GDP inflated)


# ------------------------------------------------------------------ calibration
def band_ranges() -> pd.DataFrame:
    """Residual quantiles (log RON) per headcount band from the out-of-fold predictions of the lean model."""
    oof = pd.read_parquet(PROC / "oof_band_model.parquet")
    r = oof["y"] - oof["lean"]
    big = oof["emp_band"].isin(BAND_LABELS[BAND_LABELS.index(POOL_FROM):])
    rows = {}
    for b in BAND_LABELS:
        m = oof["emp_band"] == b
        pooled = m.sum() < 100
        x = r[big] if pooled else r[m]
        rows[b] = {"lo": x.quantile(Q_LO), "med": x.quantile(0.5), "hi": x.quantile(Q_HI),
                   "n": int(m.sum()), "pooled": pooled}
    return pd.DataFrame(rows).T


def fx_rate(fx: pd.DataFrame, currency: str, period: str) -> float:
    """Units per EUR. Figures for FY2024 use the 2024 average; anything later uses 2025 (latest full year)."""
    if currency == "EUR":
        return 1.0
    year = 2024 if "2024" in str(period) and "2024/25" not in str(period) else 2025
    return float(fx.loc[(fx["currency"] == currency) & (fx["year"] == year), "per_eur"].iloc[0])


def withhold_reason(r, f, ratio: float) -> str:
    """Model revenue is shown only where the evidence says it works: small/mid local firms.
    Every sample firm with known revenue and 1k+ staff or a foreign parent was underestimated 3-60x."""
    if r["emp_band"] in BIG_BANDS or r["emp_band"] == "1k-5k" or r["emp_scope"] == "upper bound":
        return ("company has 1,000+ staff; on every such licensee with known revenue the model was 3-60x too low "
                "(Romanian training data has almost no companies this size)")
    if pd.to_numeric(f["foreign_parent"], errors="coerce") == 1:
        return ("subsidiary of a multinational; these book group sales through small local teams "
                "(back-test: Cargill RO 14x, Mercedes-Benz RO 10x too low)")
    if r["country"] != "RO" and RATIO_UNRELIABLE(ratio):
        return f"country productivity ratio ({ratio:.2f}x Romania) is not credible for this company"
    return ""


# ------------------------------------------------------------------ inputs
def model_inputs(ev: pd.DataFrame) -> pd.DataFrame:
    """One row of lean-model features per licensee (NaN where unknown; LightGBM handles missing)."""
    inp = pd.read_csv(NOTES / "sample_inputs.csv", dtype={"nace4": str})
    inp["caen4"] = inp["nace4"].str.zfill(4)
    inp["caen2"] = inp["caen4"].str[:2]
    inp["age_years"] = 2024.5 - inp["founded_year"]
    foreign = inp.set_index("id")[["caen2", "caen4", "age_years", "is_sa", "capital_city", "foreign_parent", "model_applies"]]

    comp = pd.read_parquet(PROC / "companies.parquet", columns=["cui"] + [c for c in COLS if c != "emp_band"])
    comp = comp.drop_duplicates("cui").set_index("cui")
    ro = pd.DataFrame({i: comp.loc[c] for i, c in RO_CUI.items() if c in comp.index}).T
    ro["model_applies"] = "yes"
    feats = pd.concat([foreign, ro])
    for i, why in NOT_MODELLED.items():
        feats.loc[i, "model_applies"] = "no"
    return ev[["id"]].join(feats, on="id").set_index("id")


def predict(model, cats, feats: pd.DataFrame, bands: pd.Series) -> pd.Series:
    d = feats.copy()
    d["emp_band"] = pd.Categorical(bands, categories=BAND_LABELS, ordered=True)
    for c in ("is_sa", "capital_city", "foreign_parent", "age_years"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return pd.Series(model.predict(make_X(d, COLS, cats)), index=feats.index)


# ------------------------------------------------------------------ helpers for the client sheet
def eur_m(x) -> float:
    return np.nan if pd.isna(x) else round(x / 1e6, 1)


def emp_evidence(r) -> tuple[str, str]:
    if r["status"] in ("brand",):
        return "Tier 4: no legal entity", ""
    q = r["emp_quality"]
    tier = {"reported": "Tier 1", "vendor estimate": "Tier 2", "self-reported band": "Tier 2",
            "inferred": "Inferred"}.get(q, "Tier 4")
    scope = {"E": "this entity", "E?": "probably this entity", "G": "whole group",
             "G?": "probably whole group", "upper bound": "upper bound"}.get(str(r["emp_scope"]), "")
    return tier, f"{q}; {scope}; as of {r['emp_asof']}; {r['emp_source']}".replace("; ;", ";").strip("; ")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ev = pd.read_csv(NOTES / "sample_evidence.csv", dtype={"emp_band": str, "linkedin_band_backtest": str})
    ratios = pd.read_csv(REPORTS / "productivity_ratios.csv").set_index("country")["ratio_vs_RO"]
    fx = pd.read_csv(REPORTS / "fx.csv")
    ron_per_eur = fx_rate(fx, "RON", "FY2024")  # model revenue is 2024 RON
    ranges = band_ranges()

    train = pd.read_parquet(PROC / "train.parquet")
    from band_model import emp_band
    train["emp_band"] = emp_band(train["employees"])
    model, cats = fit(train, COLS)

    feats = model_inputs(ev)
    ev = ev.set_index("id")
    bands = ev["emp_band"].where(ev["emp_band"].isin(BAND_LABELS))
    applies = feats["model_applies"].isin(["yes", "cross-check only"]) & bands.notna() & (ev["status"] == "active")
    pred = predict(model, cats, feats[applies], bands[applies])

    out = []
    for i, r in ev.iterrows():
        ratio = float(ratios.get(r["country"], np.nan)) if r["country"] != "RO" else 1.0
        flags, action, withheld_model = [], "", ""
        # ---- model estimate (EUR), if the model applies
        m_pt = m_lo = m_hi = np.nan
        if i in pred.index:
            q = ranges.loc[r["emp_band"]]
            scale = ratio / ron_per_eur
            m_pt, m_lo, m_hi = (np.exp(pred[i] + q[k]) * scale for k in ("med", "lo", "hi"))
            if r["emp_band"] in BIG_BANDS:
                flags.append("model outside its training data (Romania has 18 firms of 5k+ staff); range borrowed from the 1k+ bands")
            elif r["emp_band"] == "1k-5k":
                flags.append("1k-5k ranges held only 67% of true values in testing: treat as too narrow")
            if r["country"] != "RO" and RATIO_UNRELIABLE(ratio):
                flags.append(f"country productivity ratio ({ratio:.2f}x Romania) is unreliable for a formal-sector multinational")
        # ---- revenue ladder
        if r["status"] == "brand":
            tier, pt, lo, hi, rev_ev = "Tier 4", np.nan, np.nan, np.nan, "Not a legal entity"
            action = "Ask which legal entity signed the contract"
        elif r["status"] == "unmatched":
            tier, pt, lo, hi, rev_ev = "Tier 4", np.nan, np.nan, np.nan, "No registry match for this legal name"
            action = "Ask for the registration number on the contract"
        elif r["status"] == "defunct":
            tier, pt, lo, hi, rev_ev = "Tier 4", 0.0, np.nan, np.nan, "Company no longer operates"
            action = "Review the licence: the licensee has ceased to exist"
        elif pd.notna(r["rev_value"]):
            rate = fx_rate(fx, r["rev_currency"], r["rev_period"])
            tier, pt, lo, hi = "Tier 1", r["rev_value"] / rate, np.nan, np.nan
            rev_ev = f"{r['rev_quality']}; {r['rev_period']}; {'group' if r['rev_scope'] == 'G' else 'entity'}; {r['rev_source']}"
            if r["rev_quality"] != "reported":
                flags.append("revenue is a third-party estimate, not a filed figure")
            flags = [f for f in flags if not f.startswith(("model outside", "1k-5k ranges", "country productivity"))]
        elif i in pred.index and (why := withhold_reason(r, feats.loc[i], ratio)):
            # decision log 2026-11-15 #6: the model underestimates large / multinational firms 5-60x
            tier, pt, lo, hi = "Withheld", np.nan, np.nan, np.nan
            rev_ev = f"No defensible estimate: {why}"
            action = "Revenue needs a filed figure (local registry accounts or the group's segment reporting)"
            flags = [f for f in flags if not f.startswith(("model outside", "1k-5k ranges", "country productivity"))]
            withheld_model = f"withheld; model said {eur_m(m_lo)}-{eur_m(m_hi)} (point {eur_m(m_pt)}), not used"
        elif i in pred.index:
            tier, pt, lo, hi = "Tier 2", m_pt, m_lo, m_hi
            rev_ev = (f"model: {r['emp_band']} staff band + industry/age/legal form, learned from 685k Romanian filings, "
                      f"x{ratio:.2f} country productivity; 80% range")
            if r["emp_scope"] == "upper bound":
                pt, lo = np.nan, np.nan
                rev_ev = "upper bound only: headcount is itself an upper bound; " + rev_ev
        else:
            tier, pt, lo, hi = "Tier 4", np.nan, np.nan, np.nan
            rev_ev = NOT_MODELLED.get(i, "no reported figure and no usable headcount band")
        if str(r["emp_scope"]).startswith("G") and r["status"] == "active":
            flags.append("headcount may count the whole group, not this entity")
        if pd.notna(r["note"]):
            flags.append(str(r["note"]))
        emp_tier, emp_src = emp_evidence(r)
        cross = withheld_model or ("" if tier != "Tier 1" or np.isnan(m_pt) else (
            f"{eur_m(m_lo)}-{eur_m(m_hi)} (point {eur_m(m_pt)}); reported is "
            + ("inside" if m_lo <= pt <= m_hi else "OUTSIDE") + " the model range"))
        confidence = ("High" if tier == "Tier 1" and r["rev_quality"] == "reported" else
                      "Medium" if tier == "Tier 1" else
                      "Low" if tier == "Tier 2" else "None")
        out.append({
            "ID": i, "Licensee (as listed)": r["licensee"], "Country": r["country"], "Status": r["status"],
            "Employees": r["emp_value"], "Employee band": r["emp_band"], "Employees: tier": emp_tier,
            "Employees: evidence": emp_src,
            "Revenue EUR m": eur_m(pt), "Revenue low EUR m (P10)": eur_m(lo), "Revenue high EUR m (P90)": eur_m(hi),
            "Revenue: tier": tier, "Revenue: evidence": rev_ev, "Model cross-check EUR m": cross,
            "Confidence": confidence, "Caveats": " | ".join(flags), "Action for client": action,
        })
    est = pd.DataFrame(out)

    # ---- back-test: Romanian rows, model vs filed truth (RON, no ratio)
    bt = []
    for i in [k for k in RO_CUI if k in pred.index]:
        r = ev.loc[i]
        f = feats.loc[[i]]
        for label, band in (("true band", r["emp_band"]), ("LinkedIn band", r["linkedin_band_backtest"])):
            if pd.isna(band):
                continue
            p = predict(model, cats, f, pd.Series([band], index=[i]))[i]
            q = ranges.loc[band]
            pt, lo, hi = (np.exp(p + q[k]) for k in ("med", "lo", "hi"))
            bt.append({"id": i, "licensee": r["licensee"], "band_used": f"{label}: {band}",
                       "true_revenue_RON_m": round(r["rev_value"] / 1e6, 1), "model_RON_m": round(pt / 1e6, 1),
                       "low_RON_m": round(lo / 1e6, 1), "high_RON_m": round(hi / 1e6, 1),
                       "factor_off": round(float(np.exp(abs(np.log(pt / r["rev_value"])))), 2),
                       "truth_inside_range": bool(lo <= r["rev_value"] <= hi)})
    pd.DataFrame(bt).to_csv(REPORTS / "backtest_romania.csv", index=False)

    # ---- ratio check: foreign rows with a reported revenue and a model estimate
    rc = []
    for i in pred.index:
        r = ev.loc[i]
        if r["country"] == "RO" or pd.isna(r["rev_value"]):
            continue
        rep = r["rev_value"] / fx_rate(fx, r["rev_currency"], r["rev_period"])
        raw = np.exp(pred[i] + ranges.loc[r["emp_band"], "med"]) / ron_per_eur
        ratio = float(ratios[r["country"]])
        rc.append({"id": i, "licensee": r["licensee"], "country": r["country"], "band": r["emp_band"],
                   "reported_EUR_m": eur_m(rep), "model_no_ratio_EUR_m": eur_m(raw), "ratio": ratio,
                   "model_with_ratio_EUR_m": eur_m(raw * ratio),
                   "reported / model_with_ratio": round(rep / (raw * ratio), 2),
                   "reported / model_no_ratio": round(rep / raw, 2), "rev_quality": r["rev_quality"]})
    pd.DataFrame(rc).to_csv(REPORTS / "ratio_check.csv", index=False)

    # ---- the client file
    lic = pd.read_csv(ROOT / "data/raw/licensee_list.csv")
    rest = lic[~lic["id"].isin(est["ID"])].rename(columns={"id": "ID", "licensee_name": "Licensee (as listed)",
                                                            "country": "Country"})[["ID", "Licensee (as listed)", "Country"]]
    rest["Status"] = "not assessed in this sample"
    method = pd.DataFrame({"How to read this file": [
        "Each number climbs an evidence ladder on its own. Tier 1 = a reported figure (filing, annual report, registry). "
        "Tier 2 = an observed headcount band turned into revenue by a model. Tier 4 = no defensible number; the row says why and what would fix it.",
        "Revenue in EUR millions. Reported figures converted at ECB annual average rates (FY2024 → 2024 rate, later periods → 2025 rate).",
        "Tier 2 revenue: a model learned from 685,000 Romanian 2024 company filings (revenue vs. headcount band, industry, age, legal form, "
        "capital-city location, foreign parent), scaled by the country's output per worker relative to Romania (World Bank, 2024, market exchange rates).",
        "Ranges are 80% ranges: by design, about 1 in 5 true values falls outside. They come from the model's own out-of-sample errors per headcount band.",
        "Confidence: High = reported by the company or a registry; Medium = a third-party figure (e.g. Forbes); Low = model estimate; None = no estimate.",
        "'Model cross-check' on tier 1 rows shows what the model would have said; 'OUTSIDE' means the model would have been wrong for this company.",
        "'Withheld': the model produced a number but we do not stand behind it. On every licensee with known revenue that has 1,000+ staff or a "
        "foreign parent, the model was 3-60x too low. A missing number is safer to price on than a plausible wrong one; the model's output is "
        "still shown in the cross-check column so the decision is visible.",
        "Headcount scope matters: 'whole group' figures (e.g. LinkedIn's global page) can overstate a subsidiary by 10-70x (seen in our Romanian back-test).",
        f"{100 - len(est)} of the 100 licensees were not assessed: this file is a proof of the method on a sample covering every problem type in the list.",
    ]})
    xlsx = OUT / "licensee_size_estimates.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        est.to_excel(w, sheet_name="Estimates (sample)", index=False)
        method.to_excel(w, sheet_name="How to read this", index=False)
        rest.to_excel(w, sheet_name="Not assessed", index=False)
        for ws in w.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width * 0.9), 70)
            ws.freeze_panes = "C2"
    est.to_csv(OUT / "licensee_size_estimates.csv", index=False)

    pd.set_option("display.width", 250)
    print(est[["ID", "Licensee (as listed)", "Employees", "Revenue EUR m", "Revenue low EUR m (P10)",
               "Revenue high EUR m (P90)", "Revenue: tier", "Confidence"]].to_string(index=False))
    print("\nBACK-TEST\n", pd.DataFrame(bt).to_string(index=False))
    print("\nRATIO CHECK\n", pd.DataFrame(rc).to_string(index=False))


if __name__ == "__main__":
    main()
