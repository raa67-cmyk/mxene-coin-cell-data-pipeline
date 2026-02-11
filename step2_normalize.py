from pathlib import Path
import pandas as pd
import sys, traceback

ROOT = Path(__file__).resolve().parent
csv_path = ROOT / "data" / "raw" / "cell01.csv"
parquet_path = ROOT / "data" / "interim" / "cell01_timeseries.parquet"

print("-> running step2_normalize.py")
print("Project root:", ROOT)
print("CSV exists?", csv_path.exists(), "|", csv_path)
print("Parquet target:", parquet_path)

try:
    # 1) read + sort
    df = pd.read_csv(csv_path, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 2) standardize labels
    map_to = {"CC DISCHARGE": "CC_DIS", "CC CHARGE": "CC_CHG", "CV CHARGE": "CV"}
    df["step_type"] = df["step_type"].astype(str).str.upper().replace(map_to)

    # 3) enforce discharge-negative (flip if most discharge rows are positive)
    dis_mask = df["step_type"].str.contains("DIS", na=False)
    if dis_mask.any() and (df.loc[dis_mask, "current_a"] > 0).mean() > 0.8:
        df["current_a"] = -df["current_a"]

    # 4) write parquet
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    print("Rows:", len(df))
    print("Wrote:", parquet_path)
except Exception as e:
    print("!! ERROR:", e)
    traceback.print_exc()
    sys.exit(1)


