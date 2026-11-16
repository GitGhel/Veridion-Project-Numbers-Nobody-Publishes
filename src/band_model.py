"""Ladder tier 2: revenue given an observed headcount band, plus calibrated ranges.

Headcount is observed (LinkedIn / registry), but only as a band, so the model is trained on the band,
never the exact count. Two feature sets are compared by 5-fold CV on TRAIN only:
  full : band + every portable register feature (same as cv_models.py)
  lean : band + industry + age + legal form + capital city + foreign parent
         (drops has_web, n_caen_authorised, n_reps, any_foreign_rep, has_parent, name_group_word:
          coded from the Romanian register, and off-distribution for a foreign multinational)

Ranges come from out-of-fold residuals, per headcount band, so the band-specific bias seen in CV
(big firms under-predicted) is corrected by the band's median residual and the width reflects that
band's own spread. Honesty check: quantiles are set on a random half of the OOF rows and coverage is
measured on the other half.

Also builds the same range table for ladder tier 3 (register-only LightGBM from cv_models.py),
binned by predicted value, since no band is observed there.

Outputs
  reports/band_model_cv.csv          metrics: feature set x (ALL, revenue bands, headcount bands)
  reports/ranges_tier2.csv           per headcount band: residual quantiles, width, held-out coverage
  reports/ranges_tier3.csv           per predicted-revenue decile: same, for the register-only model
  data/processed/oof_band_model.parquet

Run: python src/band_model.py   (needs cv_models.py to have run, for tier 3)
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
PROC, REPORTS = ROOT / "data/processed", ROOT / "reports"
SEED, N_FOLDS = 42, 5
LEVEL = 0.80  # range = central 80% of residuals; a judgment call, see decision log
Q_LO, Q_HI = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2

# LinkedIn company-size bands (plus 0, which LinkedIn lacks: shells / holdings)
BAND_EDGES = [-1, 0, 1, 10, 50, 200, 500, 1000, 5000, 10000, np.inf]
BAND_LABELS = ["0", "1", "2-10", "11-50", "51-200", "201-500", "501-1k", "1k-5k", "5k-10k", "10k+"]

FEATURES = {
    "full": ["emp_band", "caen2", "caen4", "age_years", "is_sa", "capital_city", "foreign_parent",
             "has_parent", "has_web", "name_group_word", "n_caen_authorised", "n_reps", "any_foreign_rep"],
    "lean": ["emp_band", "caen2", "caen4", "age_years", "is_sa", "capital_city", "foreign_parent"],
}
CATS = ("caen2", "caen4")
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=20,
                  random_state=SEED, verbose=-1)


def emp_band(employees: pd.Series) -> pd.Series:
    return pd.cut(employees, BAND_EDGES, labels=BAND_LABELS)


def make_X(d: pd.DataFrame, cols: list[str], cats: dict) -> pd.DataFrame:
    X = d[cols].copy()
    X["emp_band"] = d["emp_band"].cat.codes.astype(float)  # ordinal: bands are ordered
    for c in CATS:
        X[c] = pd.Categorical(d[c].where(d[c].isin(cats[c])), categories=cats[c])
    return X


def fit(train: pd.DataFrame, cols: list[str]):
    cats = {c: sorted(train[c].dropna().unique()) for c in CATS}
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(make_X(train, cols, cats), train["log_revenue"])
    return model, cats


def metrics(e: np.ndarray) -> dict:
    return {"n": len(e), "median_factor_off": float(np.exp(np.median(np.abs(e)))),
            "share_within_2x": float(np.mean(np.abs(e) <= np.log(2))), "bias_log": float(np.mean(e))}


def range_table(resid: np.ndarray, groups: pd.Series, rng) -> pd.DataFrame:
    """Per group: residual quantiles (y - pred, in logs) set on half A, coverage tested on half B.
    The published quantiles use all rows; the coverage column is the honest out-of-sample check."""
    half = rng.random(len(resid)) < 0.5
    rows = []
    for g in groups.cat.categories:
        m = (groups == g).to_numpy()
        if m.sum() < 30:
            rows.append({"group": g, "n": int(m.sum())})
            continue
        r, a, b = resid[m], resid[m & half], resid[m & ~half]
        lo_a, hi_a = np.quantile(a, [Q_LO, Q_HI])
        lo, med, hi = np.quantile(r, [Q_LO, 0.5, Q_HI])
        rows.append({
            "group": g, "n": int(m.sum()),
            "resid_lo": lo, "resid_median": med, "resid_hi": hi,
            # range as multiples of the corrected point estimate, e.g. 0.4x .. 2.5x
            "low_x": float(np.exp(lo - med)), "high_x": float(np.exp(hi - med)),
            "width_factor": float(np.exp(hi - lo)),  # high end / low end
            "coverage_heldout": float(np.mean((b >= lo_a) & (b <= hi_a))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(PROC / "train.parquet")
    df["emp_band"] = emp_band(df["employees"])
    rev_band = pd.cut(df["revenue"], [0, 1e6, 1e7, 5e7, np.inf], labels=["<1m RON", "1-10m", "10-50m", ">50m"])
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(df))
    y = df["log_revenue"].to_numpy()

    oof = pd.DataFrame({"cui": df["cui"].to_numpy(), "y": y, "emp_band": df["emp_band"].to_numpy()})
    rows = []
    for name, cols in FEATURES.items():
        pred = np.empty(len(df))
        for tr, te in folds:
            model, cats = fit(df.iloc[tr], cols)
            pred[te] = model.predict(make_X(df.iloc[te], cols, cats))
        oof[name] = pred
        e = pred - y
        rows.append({"features": name, "slice": "ALL", **metrics(e)})
        for kind, grp in (("revenue", rev_band), ("headcount", df["emp_band"])):
            for g in grp.cat.categories:
                m = (grp == g).to_numpy()
                if m.any():
                    rows.append({"features": name, "slice": f"{kind} {g}", **metrics(e[m])})
        r = rows[-(len(rev_band.cat.categories) + df["emp_band"].nunique()) - 1]
        print(f"{name:5} typical miss {r['median_factor_off']:.2f}x  within 2x {r['share_within_2x']:.1%}")

    pd.DataFrame(rows).round(4).to_csv(REPORTS / "band_model_cv.csv", index=False)
    oof.to_parquet(PROC / "oof_band_model.parquet", index=False)

    rng = np.random.default_rng(SEED)
    t2 = range_table(y - oof["lean"].to_numpy(), df["emp_band"], rng)
    t2.round(4).to_csv(REPORTS / "ranges_tier2.csv", index=False)

    t3_oof = pd.read_parquet(PROC / "oof_revenue.parquet")
    p3 = t3_oof["4_lgbm"].to_numpy()
    deciles = pd.qcut(p3, 10, labels=[f"D{i}" for i in range(1, 11)])
    t3 = range_table(t3_oof["y"].to_numpy() - p3, pd.Series(deciles), rng)
    edges = np.quantile(p3, np.linspace(0, 1, 11))
    t3.insert(1, "pred_revenue_from_RON", np.exp(edges[:-1]).round(-3))
    t3.insert(2, "pred_revenue_to_RON", np.exp(edges[1:]).round(-3))
    t3.round(4).to_csv(REPORTS / "ranges_tier3.csv", index=False)
    print("wrote band_model_cv.csv, ranges_tier2.csv, ranges_tier3.csv")


if __name__ == "__main__":
    main()
