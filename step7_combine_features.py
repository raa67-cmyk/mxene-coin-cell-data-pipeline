from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
cap_path  = ROOT / "data" / "processed" / "cell01_features_step4.csv"
ener_path = ROOT / "data" / "processed" / "cell01_energy_step5.csv"
ir_path   = ROOT / "data" / "processed" / "cell01_ir_step6.csv"

out_csv    = ROOT / "data" / "processed" / "cell01_features_combined.csv"
out_parquet= ROOT / "data" / "processed" / "cell01_features_combined.parquet"

print("Reading:")
print(" -", cap_path)
print(" -", ener_path)
print(" -", ir_path)

cap  = pd.read_csv(cap_path)
ener = pd.read_csv(ener_path)
ir   = pd.read_csv(ir_path)

feat = cap.merge(ener, on="cycle_index", how="left") \
          .merge(ir,   on="cycle_index", how="left")

# ensure normalized capacity column exists
if "q_norm" not in feat.columns:
    feat["q_norm"] = feat["Q_dis_Ah"] / feat["Q_dis_Ah"].iloc[0]

# save
out_csv.parent.mkdir(parents=True, exist_ok=True)
feat.to_csv(out_csv, index=False)
try:
    feat.to_parquet(out_parquet, index=False)
except Exception as e:
    print("(Parquet save skipped:", e, ")")

print("\nCombined features:")
print(feat)
print("\nWrote:", out_csv)
