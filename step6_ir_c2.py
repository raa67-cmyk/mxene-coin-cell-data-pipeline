from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
parquet_path = ROOT / "data" / "interim" / "cell01_timeseries.parquet"
out_csv = ROOT / "data" / "processed" / "cell01_ir_step6.csv"

RATED_AH = 3.0
TARGET = 0.5 * RATED_AH  # C/2

df = pd.read_parquet(parquet_path)

def ir_c_over_2(g: pd.DataFrame, target: float) -> float:
    dis = g[g["step_type"].astype(str).str.contains("DIS")]
    if dis.empty:
        return np.nan
    # pick index where |I| is closest to target
    idx = (dis["current_a"].abs() - target).abs().idxmin()
    # small windows (toy data); for real data use wider windows (e.g., w=5–10)
    w = 1
    pre = dis.loc[max(dis.index.min(), idx - w): idx - 1]
    post = dis.loc[idx: min(idx + w, dis.index.max())]
    if pre.empty or post.empty:
        return np.nan
    dV = post["voltage_v"].median() - pre["voltage_v"].median()
    dI = post["current_a"].median() - pre["current_a"].median()
    if dI == 0 or pd.isna(dI):
        return np.nan
    return abs(dV / dI)

rows = []
for k, g in df.groupby("cycle_index", sort=True):
    rows.append({"cycle_index": int(k), "IR_C2_ohm": ir_c_over_2(g, TARGET)})

out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print(pd.read_csv(out_csv))
