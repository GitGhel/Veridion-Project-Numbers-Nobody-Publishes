"""Build the modelling dataset: 2024 financial filings joined to the trade register.

Outputs
  data/processed/companies.parquet  all filers with features + flags (pre-split)
  data/processed/train.parquet      80% of the modelling set
  data/processed/holdout.parquet    20%, frozen; only evaluate.py may read it
  reports/funnel.csv                row counts after each exclusion step

Run: python src/build_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, REPORTS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "reports"
SEED = 42
REF_DATE = pd.Timestamp("2024-12-31")  # filings describe fiscal year 2024

# ONRC status codes (nomenclator_stari_firma.csv). A company is "alive" if it is
# "in functiune" (1048) and carries none of the codes below.
DEAD_OR_DYING = {
    "1084",  # radiata (deregistered)
    "1049", "1076", "1094", "1098", "1100", "1109", "1111", "1113", "1120",  # dissolution variants
    "1052",  # lichidare
    "1070", "1139",  # faliment (1139 is newer; always co-occurs with 1070)
    "1073", "1083", "1105", "1107", "1121",  # insolvency variants
    "1074",  # intrerupere temporara de activitate (dormant by declaration)
}
COMPANY_FORMS = {"SRL", "SA", "SNC", "SCS", "SCA"}  # commercial companies; excludes PFA/II/IF/PF/AF, coops, RA
HOLDING_CAEN = {"6420"}  # turnover of a pure holding is dividends, not sales


def load_filings() -> pd.DataFrame:
    cols = {"CUI": "cui", "CAEN": "caen", "I13": "revenue", "I20": "employees",
            "I1": "fixed_assets", "I10": "equity"}
    parts = []
    for fname, ftype in [("web_uu_an2024.txt", "abbreviated"), ("web_bl_bs_sl_an2024.txt", "full")]:
        d = pd.read_csv(RAW / fname, usecols=list(cols), dtype={"CUI": "Int64", "CAEN": "string"})
        d = d.rename(columns=cols)
        # a blank I20 behaves exactly like 0 (same revenue distribution), so it is an unrecorded zero
        d["employees"] = d["employees"].fillna(0)
        d["filing_type"] = ftype  # diagnostic only, NEVER a model feature (it is a function of size)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def load_register() -> pd.DataFrame:
    """One row per CUI. A CUI can have several registrations (moves between counties
    re-register the company); keep the alive one, else the most recent, and take
    company age from the earliest registration."""
    reg = pd.read_csv(
        RAW / "od_firme.csv", sep="^", dtype=str, encoding="utf-8-sig",
        usecols=["DENUMIRE", "CUI", "COD_INMATRICULARE", "DATA_INMATRICULARE", "FORMA_JURIDICA",
                 "ADR_JUDET", "WEB", "TARA_FIRMA_MAMA"],
    )
    reg.columns = ["name", "cui", "reg_code", "reg_date", "legal_form", "county", "web", "parent_country"]
    reg["cui"] = pd.to_numeric(reg["cui"], errors="coerce").astype("Int64")
    reg = reg[reg["cui"].notna() & (reg["cui"] > 0)]
    # since the Sept-2024 ONRC format change, dates carry a time: "01/10/2024 11:11:17"
    reg["reg_date"] = pd.to_datetime(reg["reg_date"].str[:10], format="%d/%m/%Y", errors="coerce")

    status = pd.read_csv(RAW / "od_stare_firma.csv", sep="^", dtype=str, encoding="utf-8-sig")
    status.columns = ["reg_code", "code"]
    codes = status.groupby("reg_code")["code"].agg(set)
    alive = codes.map(lambda s: "1048" in s and not (s & DEAD_OR_DYING))
    reg["alive"] = reg["reg_code"].map(alive).fillna(False).astype(bool)
    reg["deregistered"] = reg["reg_code"].map(codes.map(lambda s: "1084" in s)).fillna(False).astype(bool)

    first_reg = reg.groupby("cui")["reg_date"].min().rename("first_reg_date")
    reg = (reg.sort_values(["cui", "alive", "reg_date"], ascending=[True, False, False])
              .drop_duplicates("cui").merge(first_reg, on="cui"))
    return reg


def register_side_tables(reg_codes: pd.Index) -> pd.DataFrame:
    """Counts per registration: authorised activity codes and legal representatives.
    Personal data in the representatives file is reduced to counts and never stored."""
    caen = pd.read_csv(RAW / "od_caen_autorizat.csv", sep="^", dtype=str, encoding="utf-8-sig")
    caen.columns = ["reg_code", "caen", "ver"]
    caen = caen[caen["reg_code"].isin(reg_codes)]
    # count codes in the latest CAEN version a firm has, so rev2->rev3 recoding is not double-counted
    caen["ver"] = pd.to_numeric(caen["ver"], errors="coerce")
    caen = caen[caen["ver"] == caen.groupby("reg_code")["ver"].transform("max")]
    n_caen = caen.groupby("reg_code")["caen"].nunique().rename("n_caen_authorised")

    reps = pd.read_csv(RAW / "od_reprezentanti_legali.csv", sep="^", dtype=str, encoding="utf-8-sig",
                       usecols=["COD_INMATRICULARE", "CALITATE", "TARA_NASTERE"])
    reps.columns = ["reg_code", "role", "birth_country"]
    reps = reps[reps["reg_code"].isin(reg_codes)]
    reps["foreign"] = reps["birth_country"].notna() & ~reps["birth_country"].isin(["România", "Romania"])
    rep_feats = reps.groupby("reg_code").agg(n_reps=("role", "size"), any_foreign_rep=("foreign", "max"))
    return pd.concat([n_caen, rep_feats], axis=1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only 'portable' features: things observable for a company in any country."""
    df["caen4"] = df["caen"].str.zfill(4)
    df["caen2"] = df["caen4"].str[:2]
    df["age_years"] = ((REF_DATE - df["first_reg_date"]).dt.days / 365.25).clip(lower=0)
    df["is_sa"] = (df["legal_form"] == "SA").astype(int)  # public-limited form (AG / plc / S.A.)
    df["capital_city"] = (df["county"] == "Bucureşti").astype(int)
    df["foreign_parent"] = (df["parent_country"].notna() & (df["parent_country"] != "România")).astype(int)
    df["has_parent"] = df["parent_country"].notna().astype(int)
    df["has_web"] = df["web"].notna().astype(int)
    name = df["name"].fillna("").str.upper()
    df["name_group_word"] = name.str.contains(r"\b(?:HOLDING|GROUP|GRUP|INTERNATIONAL)\b").astype(int)
    df["n_caen_authorised"] = df["n_caen_authorised"].fillna(0)
    df["n_reps"] = df["n_reps"].fillna(0)
    df["any_foreign_rep"] = df["any_foreign_rep"].fillna(False).astype(int)
    df["log_revenue"] = np.log(df["revenue"].where(df["revenue"] > 0))
    df["log1p_employees"] = np.log1p(df["employees"].clip(lower=0))
    return df


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    funnel = []
    step = lambda label, d: funnel.append({"step": label, "rows": len(d)}) or print(f"{len(d):>9,}  {label}")

    fil = load_filings()
    step("2024 filings, abbreviated + full balance sheet", fil)
    fil = fil.drop_duplicates("cui", keep=False)
    step("drop CUIs that appear in both files / twice (ambiguous)", fil)

    reg = load_register()
    df = fil.merge(reg, on="cui", how="inner")
    step("matched to trade register by CUI", df)
    df = df.merge(register_side_tables(pd.Index(df["reg_code"])), left_on="reg_code",
                  right_index=True, how="left")
    df = add_features(df)
    df.to_parquet(PROC / "companies.parquet", index=False)

    df = df[df["legal_form"].isin(COMPANY_FORMS)]
    step(f"legal form in {sorted(COMPANY_FORMS)}", df)
    df = df[df["revenue"] > 0]
    step("net turnover > 0 (drops dormant shells)", df)
    df = df[df["employees"] >= 0]
    step("employee count non-negative", df)
    df = df[~df["caen4"].isin(HOLDING_CAEN)]
    step("not a pure holding (CAEN 6420)", df)
    df = df[df["first_reg_date"].notna()]
    step("registration date known", df)

    rng = np.random.default_rng(SEED)
    is_holdout = rng.random(len(df)) < 0.20
    df[~is_holdout].to_parquet(PROC / "train.parquet", index=False)
    df[is_holdout].to_parquet(PROC / "holdout.parquet", index=False)
    print(f"train {(~is_holdout).sum():,} / holdout {is_holdout.sum():,} (holdout untouched until evaluation)")
    pd.DataFrame(funnel).to_csv(REPORTS / "funnel.csv", index=False)


if __name__ == "__main__":
    main()
