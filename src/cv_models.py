"""Compare size models by 5-fold cross-validation on the TRAIN split only.

Ladder (each must beat the one before to earn its complexity):
  1. global median            : predicts the same number for everyone
  2. industry median          : median of the target within CAEN 4-digit (fallback 2-digit, then global)
  3. OLS on logs              : industry (CAEN 2-digit dummies) + portable numeric features
  4. LightGBM                 : same features + CAEN 4-digit, non-linear, interactions

Revenue (log RON) and employees (log1p) are modelled separately.

Outputs
  reports/cv_metrics.csv                     metrics per model x target, overall and by true-size band
  data/processed/oof_<target>.parquet        out-of-fold predictions (input for the prediction ranges)

Run: python src/cv_models.py
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC, REPORTS = ROOT / "data/processed", ROOT / "reports"
SEED, N_FOLDS = 42, 5

TARGETS = {"revenue": "log_revenue", "employees": "log1p_employees"}
BINARY = ["is_sa", "capital_city", "foreign_parent", "has_parent", "has_web", "name_group_word", "any_foreign_rep"]
COUNTS = ["age_years", "n_caen_authorised", "n_reps"]  # log1p-transformed for OLS (skewed)


# ---------------------------------------------------------------- models
def fit_predict_global(tr, te, y):
    return np.full(len(te), tr[y].median())


def fit_predict_industry(tr, te, y):
    g = tr[y].median()
    m4 = tr.groupby("caen4")[y].agg(["median", "size"])
    m2 = tr.groupby("caen2")[y].median()
    m4 = m4.loc[m4["size"] >= 20, "median"]  # thin 4-digit cells are noise; fall back to 2-digit
    return te["caen4"].map(m4).fillna(te["caen2"].map(m2)).fillna(g).to_numpy()


def _ols_X(d):
    X = d[BINARY + ["caen2"]].copy()
    for c in COUNTS:
        X[c] = np.log1p(d[c])
    return X


def fit_predict_ols(tr, te, y):
    pre = ColumnTransformer([
        ("ind", OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["caen2"]),
        ("num", StandardScaler(), COUNTS),
    ], remainder="passthrough")
    # Ridge with a tiny penalty = OLS, but stable with ~85 dummies
    model = make_pipeline(pre, Ridge(alpha=1.0))
    model.fit(_ols_X(tr), tr[y])
    return model.predict(_ols_X(te))


def _lgb_X(d, cats):
    X = d[BINARY + COUNTS].copy()
    for c in ("caen2", "caen4"):
        X[c] = pd.Categorical(d[c].where(d[c].isin(cats[c])), categories=cats[c])  # unseen code -> NaN
    return X


def fit_predict_lgbm(tr, te, y):
    cats = {c: sorted(tr[c].dropna().unique()) for c in ("caen2", "caen4")}
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=20,
        random_state=SEED, verbose=-1,
    )
    model.fit(_lgb_X(tr, cats), tr[y])
    return model.predict(_lgb_X(te, cats))


MODELS = {
    "1_global_median": fit_predict_global,
    "2_industry_median": fit_predict_industry,
    "3_ols": fit_predict_ols,
    "4_lgbm": fit_predict_lgbm,
}


# ---------------------------------------------------------------- metrics
def metrics(y, p):
    e = p - y
    return {
        "n": len(y),
        "rmse_log": np.sqrt(np.mean(e**2)),
        "mae_log": np.mean(np.abs(e)),
        # typical multiplicative miss: 1.0 = perfect, 2.0 = off by a factor of two
        "median_factor_off": float(np.exp(np.median(np.abs(e)))),
        "share_within_2x": np.mean(np.abs(e) <= np.log(2)),
        "bias_log": np.mean(e),  # >0 = overestimates; watch this per size band
        # undefined when the band has no variance (e.g. everyone has 0 employees)
        "r2": 1 - np.sum(e**2) / ss if (ss := np.sum((y - y.mean()) ** 2)) > 0 else np.nan,
    }


def size_band(target, d):
    """True-size bands. Licensees skew large, so the top bands matter more than the average."""
    if target == "revenue":
        edges, labels = [0, 1e6, 1e7, 5e7, np.inf], ["<1m RON", "1-10m", "10-50m", ">50m"]
        return pd.cut(d["revenue"], edges, labels=labels)
    edges, labels = [-1, 0, 9, 49, 249, np.inf], ["0", "1-9", "10-49", "50-249", "250+"]
    return pd.cut(d["employees"], edges, labels=labels)


def main() -> None:
    df = pd.read_parquet(PROC / "train.parquet")
    print(f"train rows: {len(df):,}")
    folds = list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(df))
    rows = []
    for target, y in TARGETS.items():
        oof = pd.DataFrame({"cui": df["cui"].to_numpy(), "y": df[y].to_numpy()})
        for name, fn in MODELS.items():
            pred = np.empty(len(df))
            for k, (tr_idx, te_idx) in enumerate(folds):
                pred[te_idx] = fn(df.iloc[tr_idx], df.iloc[te_idx], y)
            oof[name] = pred
            band = size_band(target, df)
            rows.append({"target": target, "model": name, "band": "ALL", **metrics(oof["y"].to_numpy(), pred)})
            for b in band.cat.categories:
                m = (band == b).to_numpy()
                rows.append({"target": target, "model": name, "band": b, **metrics(oof["y"].to_numpy()[m], pred[m])})
            r = rows[-1 - len(band.cat.categories)]
            print(f"{target:9} {name:18} R2={r['r2']:.3f}  median x-off={r['median_factor_off']:.2f}  "
                  f"within 2x={r['share_within_2x']:.1%}")
        oof.to_parquet(PROC / f"oof_{target}.parquet", index=False)

    out = pd.DataFrame(rows).round(4)
    out.to_csv(REPORTS / "cv_metrics.csv", index=False)
    print(f"\nwrote {REPORTS / 'cv_metrics.csv'}")


if __name__ == "__main__":
    main()
