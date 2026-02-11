from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
parquet_path = ROOT / "data" / "interim" / "cell01_timeseries.parquet"
out_csv = ROOT / "data" / "processed" / "cell01_energy_step5.csv"

df = pd.read_parquet(parquet_path)

def cycle_energy_wh(g: pd.DataFrame) -> float:
    dis = g[g["step_type"].astype(str).str.contains("DIS")]
    if dis.shape[0] < 2:
        return np.nan
    # convert timestamps to seconds (float)
    t = dis["timestamp"].astype("int64").to_numpy() / 1e9  # seconds
    p = (dis["voltage_v"] * dis["current_a"]).to_numpy()   # watts
    e_ws = np.trapz(p, t)                                  # watt-seconds
    return abs(e_ws) / 3600.0                              # watt-hours (absolute)

rows = []
for k, g in df.groupby("cycle_index", sort=True):
    rows.append({"cycle_index": int(k), "E_dis_Wh": cycle_energy_wh(g)})

out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print(pd.read_csv(out_csv))
