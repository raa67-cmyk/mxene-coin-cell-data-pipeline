from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
parquet_path = ROOT / "data" / "interim" / "cell01_timeseries.parquet"
out_csv = ROOT / "data" / "processed" / "cell01_features_step4.csv"

df = pd.read_parquet(parquet_path)

rows = []
for k, g in df.groupby("cycle_index", sort=True):
    q_dis = pd.to_numeric(g["discharge_ah"], errors="coerce").dropna().iloc[-1]
    q_chg = pd.to_numeric(g["charge_ah"], errors="coerce").dropna().iloc[-1]
    ce = q_dis / q_chg if q_chg not in [0, np.nan] else np.nan
    rows.append({"cycle_index": int(k), "Q_dis_Ah": q_dis, "Q_chg_Ah": q_chg, "CE": ce})

feat = pd.DataFrame(rows).sort_values("cycle_index")
feat["q_norm"] = feat["Q_dis_Ah"] / feat["Q_dis_Ah"].iloc[0]

out_csv.parent.mkdir(parents=True, exist_ok=True)
feat.to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print(feat)

