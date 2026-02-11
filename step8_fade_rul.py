from pathlib import Path
import pandas as pd
import numpy as np
import math

ROOT = Path(__file__).resolve().parent
feat_path = ROOT / "data" / "processed" / "cell01_features_combined.csv"
summary_path = ROOT / "data" / "processed" / "cell01_summary.csv"

df = pd.read_csv(feat_path).sort_values("cycle_index")

# Use q_norm vs cycle_index for a simple linear fit: q_norm ~ m*n + b
x = df["cycle_index"].to_numpy(dtype=float)
y = df["q_norm"].to_numpy(dtype=float)

# need at least 2 points to fit a line
if len(x) < 2:
    raise SystemExit("Not enough cycles to fit a line. Add more data.")

m, b = np.polyfit(x, y, 1)  # slope, intercept
slope_pct_per_cycle = m * 100.0

# cycles to 80% (EOL fraction)
eol = 0.80
n_eol = math.nan if m == 0 else (eol - b) / m

summary = {
    "Q0_Ah": float(df["Q_dis_Ah"].iloc[0]),
    "fade_slope_pct_per_cycle": float(slope_pct_per_cycle),
    "cycles_to_80pct": float(n_eol) if not math.isnan(n_eol) else None
}

pd.DataFrame([summary]).to_csv(summary_path, index=False)

print("Linear fit: q_norm ≈ m*n + b")
print(f"  m (slope) = {m:.6f}  → {slope_pct_per_cycle:.4f}% per cycle")
print(f"  b (intercept) = {b:.6f}")
print(f"Estimated cycles to 80%: {n_eol:.2f}")
print("Wrote summary:", summary_path)
