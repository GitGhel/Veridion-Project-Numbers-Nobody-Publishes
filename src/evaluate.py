"""Final, one-time evaluation on the frozen holdout (20% of filers, untouched until now).

Fits the chosen tier-2 model (lean features, see decision log) on the full train split, predicts the holdout, and
checks the 80% ranges that were calibrated on train out-of-fold residuals. Also re-scores the register-only model
(tier 3) for comparison. Nothing is tuned here: this only measures.

Output: reports/holdout_eval.csv
Run: python src/evaluate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from band_model import BAND_LABELS, FEATURES, emp_band, fit, make_X  # noqa: E402
from cv_models import fit_predict_lgbm  # noqa: E402
from ladder import band_ranges  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROC, REPORTS = ROOT / "data/processed", ROOT / "reports"


def main() -> None:
    train = pd.read_parquet(PROC / "train.parquet")
    hold = pd.read_parquet(PROC / "holdout.parquet")
    for d in (train, hold):
        d["emp_band"] = emp_band(d["employees"])
    y = hold["log_revenue"].to_numpy()

    model, cats = fit(train, FEATURES["lean"])
    p2 = model.predict(make_X(hold, FEATURES["lean"], cats))
    p3 = fit_predict_lgbm(train, hold, "log_revenue")
    rng = band_ranges()

    rows = []
    groups = [("ALL", np.ones(len(hold), bool))] + [(b, (hold["emp_band"] == b).to_numpy()) for b in BAND_LABELS]
    for name, m in groups:
        if m.sum() == 0:
            continue
        e2, e3 = p2[m] - y[m], p3[m] - y[m]
        q = rng.loc[hold.loc[m, "emp_band"].astype(str)]
        inside = (y[m] >= p2[m] + q["lo"].to_numpy()) & (y[m] <= p2[m] + q["hi"].to_numpy())
        rows.append({
            "headcount_band": name, "n": int(m.sum()),
            "tier2_typical_miss_x": round(float(np.exp(np.median(np.abs(e2)))), 2),
            "tier2_within_2x": round(float(np.mean(np.abs(e2) <= np.log(2))), 3),
            "tier2_bias_log": round(float(np.mean(e2)), 3),
            "tier2_range80_coverage": round(float(np.mean(inside)), 3),
            "tier3_typical_miss_x": round(float(np.exp(np.median(np.abs(e3)))), 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "holdout_eval.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
