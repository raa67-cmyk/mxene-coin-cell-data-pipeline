from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
combined_path = ROOT / "data" / "processed" / "cell01_features_combined.csv"
dqdv_path     = ROOT / "data" / "processed" / "cell01_dqdv_step9.csv"
out_path      = ROOT / "data" / "processed" / "cell01_features_full.csv"

feat  = pd.read_csv(combined_path)
dqdv  = pd.read_csv(dqdv_path)

full = feat.merge(dqdv, on="cycle_index", how="left")
full.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(full)
