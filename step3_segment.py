from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
parquet_path = ROOT / "data" / "interim" / "cell01_timeseries.parquet"

df = pd.read_parquet(parquet_path)

for k, g in df.groupby("cycle_index", sort=True):
    print(f"\n=== cycle {int(k)} | rows: {len(g)} ===")
    print(g[["timestamp","step_type","current_a","voltage_v","charge_ah","discharge_ah"]].head(3))
