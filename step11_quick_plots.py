from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
full_path = ROOT / "data" / "processed" / "cell01_features_full.csv"
df = pd.read_csv(full_path)

plt.figure()
plt.plot(df["cycle_index"], df["Q_dis_Ah"], marker="o")
plt.xlabel("Cycle"); plt.ylabel("Discharge Capacity (Ah)")
plt.title("Capacity vs Cycle"); plt.grid(True); plt.tight_layout()
plt.savefig(ROOT / "data" / "processed" / "plot_capacity.png")

plt.figure()
plt.plot(df["cycle_index"], df["CE"], marker="o")
plt.xlabel("Cycle"); plt.ylabel("Coulombic Efficiency")
plt.title("CE vs Cycle"); plt.grid(True); plt.tight_layout()
plt.savefig(ROOT / "data" / "processed" / "plot_ce.png")

print("Saved plots to data/processed/")
