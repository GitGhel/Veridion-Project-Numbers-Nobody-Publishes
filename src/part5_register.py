"""Part 5: for how many companies in the Romanian register could the tier-2 model produce an estimate we would use?

Lines drawn (rough, stated as such):
  register        od_firme.csv, one row per CUI (a CUI can have several registrations; see build_dataset.py)
  filtered        ALIVE (status 1048 and no dissolution / liquidation / insolvency / deregistration /
                  temporary-inactivity code; decision log 2026-11-11) AND a commercial company form (SRL, SA, SNC,
                  SCS, SCA), the population the model learned from. Denominator for the share.
  inside the line the model needs a headcount band and an industry; we show its output only where it held up
                  (decision log #6). So: filed 2024 accounts (headcount + main activity code observed; the
                  register itself has neither) AND < 1,000 staff AND no foreign parent (TARA_FIRMA_MAMA)
                  AND not finance/insurance (NACE 64-66: revenue there is not turnover)
Output: reports/part5_register.csv
Run: python src/part5_register.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import COMPANY_FORMS, load_filings, load_register  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NACE2 = {  # names for the divisions that come out on top (NACE Rev.2 / CAEN Rev.2)
    "41": "Construction of buildings", "43": "Specialised construction", "45": "Motor vehicle trade & repair",
    "46": "Wholesale trade", "47": "Retail trade", "49": "Land transport", "56": "Food & beverage service",
    "62": "Computer programming", "68": "Real estate", "69": "Legal & accounting", "70": "Head offices & consultancy",
    "71": "Architecture & engineering", "86": "Human health",
}


def main() -> None:
    reg = load_register()
    fil = load_filings().drop_duplicates("cui", keep=False)
    fil["caen2"] = fil["caen"].str.zfill(4).str[:2]

    steps = [("register: unique CUIs", len(reg))]
    filt = reg[reg["alive"] & reg["legal_form"].isin(COMPANY_FORMS)]
    steps.append(("filtered: alive + company form (denominator)", len(filt)))
    d = filt.merge(fil[["cui", "employees", "caen2", "revenue"]], on="cui", how="inner")
    steps.append(("... filed 2024 accounts (headcount + main activity observed)", len(d)))
    d = d[d["employees"] < 1000]
    steps.append(("... fewer than 1,000 staff", len(d)))
    d = d[d["parent_country"].isna() | (d["parent_country"] == "România")]
    steps.append(("... no foreign parent", len(d)))
    d = d[~d["caen2"].isin(["64", "65", "66"])]
    steps.append(("INSIDE THE LINE: ... not finance / insurance", len(d)))

    share = len(d) / len(filt)
    top = d["caen2"].value_counts().head(3)
    dormant = int((d["revenue"] <= 0).sum())

    out = pd.DataFrame(steps, columns=["step", "companies"]).astype({"companies": object})
    out.loc[len(out)] = ["share of filtered register", f"{share:.1%}"]
    for code, n in top.items():
        out.loc[len(out)] = [f"top industry: NACE {code} {NACE2.get(code, '')}", n]
    out.loc[len(out)] = ["(inside the line but zero 2024 turnover: dormant, model would still output a number)", dormant]
    out.to_csv(REPORTS / "part5_register.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
