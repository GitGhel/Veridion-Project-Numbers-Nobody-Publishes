"""Output per worker by country, relative to Romania: the scaling factor for foreign tier-2 revenue.

ratio_c = (GDP_c / employed_c) / (GDP_RO / employed_RO), at current US$ (market exchange rates, not PPP:
the client prices contracts in money, so we want booked revenue, not real output; see decision log 2026-11-14).

Sources (free, no key):
  World Bank WDI  NY.GDP.MKTP.CD  GDP, current US$
                  SL.TLF.TOTL.IN  labour force, total
                  SL.UEM.TOTL.ZS  unemployment, % of labour force (ILO modelled)
                  employed = labour force x (1 - unemployment)
  ECB             EXR A.<CUR>.EUR.SP00.A  annual averages per EUR, to report revenue in EUR

Year: 2024 to match the filings; falls back to the latest earlier year available, and says so.

Outputs
  reports/productivity_ratios.csv   one row per country on the licensee list (+ RO)
  reports/fx.csv                    ECB annual averages (per EUR) for 2024 and 2025
Run: python src/productivity.py
"""
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, REPORTS = ROOT / "data/raw", ROOT / "reports"
YEAR = 2024
INDICATORS = {"gdp_usd": "NY.GDP.MKTP.CD", "labour_force": "SL.TLF.TOTL.IN", "unemp_pct": "SL.UEM.TOTL.ZS"}


def get_json(url: str, tries: int = 4):
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    for attempt in range(tries):  # the World Bank API is slow and times out now and then
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except (TimeoutError, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def wb(indicator: str, countries: list[str]) -> pd.DataFrame:
    url = (f"https://api.worldbank.org/v2/country/{';'.join(countries)}/indicator/{indicator}"
           f"?format=json&date={YEAR - 5}:{YEAR}&per_page=1000")
    rows = get_json(url)[1] or []
    d = pd.DataFrame([{"iso2": r["country"]["id"], "year": int(r["date"]), "value": r["value"]} for r in rows])
    return d.dropna(subset=["value"])


def latest(d: pd.DataFrame) -> pd.DataFrame:
    return d.sort_values("year").groupby("iso2").tail(1).set_index("iso2")


def main() -> None:
    lic = pd.read_csv(RAW / "licensee_list.csv")
    countries = sorted(set(lic["country"].dropna()) | {"RO"})

    out = pd.DataFrame(index=countries)
    for name, ind in INDICATORS.items():
        d = latest(wb(ind, countries))
        out[name], out[f"{name}_year"] = d["value"], d["year"]
    out["employed"] = out["labour_force"] * (1 - out["unemp_pct"] / 100)
    out["gdp_per_worker_usd"] = out["gdp_usd"] / out["employed"]
    out["ratio_vs_RO"] = out["gdp_per_worker_usd"] / out.loc["RO", "gdp_per_worker_usd"]
    out["data_year"] = out[[c for c in out if c.endswith("_year")]].min(axis=1).astype("Int64")
    out["note"] = out["data_year"].map(lambda y: "" if y == YEAR else f"latest available: {y}")
    keep = ["gdp_per_worker_usd", "ratio_vs_RO", "data_year", "note"]
    out = out[keep].rename_axis("country").sort_values("ratio_vs_RO", ascending=False)
    out.round({"gdp_per_worker_usd": 0, "ratio_vs_RO": 3}).to_csv(REPORTS / "productivity_ratios.csv")
    print(out.round(2).to_string())

    # ECB annual averages, units of currency per EUR, for every currency the sample evidence uses
    rows = []
    for cur in ("RON", "USD", "GBP", "INR", "SEK"):
        j = get_json(f"https://data-api.ecb.europa.eu/service/data/EXR/A.{cur}.EUR.SP00.A?format=jsondata")
        series = next(iter(j["dataSets"][0]["series"].values()))["observations"]
        years = j["structure"]["dimensions"]["observation"][0]["values"]
        for k, v in series.items():
            y = int(years[int(k)]["id"])
            if y in (YEAR, YEAR + 1):
                rows.append({"currency": cur, "year": y, "per_eur": v[0], "source": f"ECB EXR A.{cur}.EUR.SP00.A"})
    fx = pd.DataFrame(rows).sort_values(["currency", "year"])
    fx.to_csv(REPORTS / "fx.csv", index=False)
    print(fx.to_string(index=False))

if __name__ == "__main__":
    main()
