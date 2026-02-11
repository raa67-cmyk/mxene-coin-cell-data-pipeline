# step12_qc.py — quick Quality Control checks for one processed cell
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True, help="Cell ID used when running pipeline.py (e.g., ARBIN01)")
    args = p.parse_args()

    feat_path = ROOT / "data" / "processed" / f"{args.cell}_features_full.csv"
    if not feat_path.exists():
        raise SystemExit(f"Missing file: {feat_path} — run pipeline.py first with --cell {args.cell}")

    df = pd.read_csv(feat_path).sort_values("cycle_index")
    msgs = []

    # 1) Basic integrity
    if df["cycle_index"].isna().any():
        msgs.append("cycle_index has NaNs.")
    if df["Q_dis_Ah"].isna().any():
        msgs.append("Q_dis_Ah has NaNs (capacity missing).")

    # 2) CE sanity (expect ~1.0)
    ce_out = df["CE"].dropna()
    if not ce_out.empty:
        ce_min, ce_max = ce_out.min(), ce_out.max()
        if (ce_min < 0.95) or (ce_max > 1.05):
            msgs.append(f"CE outside [0.95,1.05]: min={ce_min:.3f}, max={ce_max:.3f}")

    # 3) Capacity trend (should not rise a lot over cycles)
    if len(df) >= 2:
        if (df["Q_dis_Ah"].iloc[-1] - df["Q_dis_Ah"].iloc[0]) > 0.02 * df["Q_dis_Ah"].iloc[0]:
            msgs.append("Capacity increased >2% from first to last cycle (unexpected).")

    # 4) Energy positive
    if "E_dis_Wh" in df.columns:
        if (df["E_dis_Wh"] <= 0).any():
            msgs.append("Some discharge energies <= 0 Wh.")

    # 5) IR @ C/2 reasonable (toy thresholds; tune for your cell)
    if "IR_C2_ohm" in df.columns:
        ir = df["IR_C2_ohm"].dropna()
        if not ir.empty:
            if (ir <= 0).any():
                msgs.append("IR_C2 has non-positive values.")
            if ir.median() > 0.2:
                msgs.append(f"Median IR_C2 seems high: {ir.median():.3f} Ω")

    # 6) dQ/dV peak shift not wild
    if "dQdV_shift_mV" in df.columns:
        shifts = df["dQdV_shift_mV"].dropna().abs()
        if not shifts.empty and shifts.max() > 200:
            msgs.append(f"dQ/dV peak shift > 200 mV detected (max {shifts.max():.1f} mV)")

    # 7) Print summary
    print(f"\nQC for {args.cell}")
    print(f"Rows: {len(df)} | Cycles: {df['cycle_index'].nunique()}")
    if "CE" in df.columns and not df['CE'].dropna().empty:
        print(f"CE range: {df['CE'].min():.3f} → {df['CE'].max():.3f}")
    if "IR_C2_ohm" in df.columns and not df['IR_C2_ohm'].dropna().empty:
        print(f"IR_C2 median: {df['IR_C2_ohm'].median():.4f} Ω")

    if msgs:
        print("\nWARNINGS:")
        for m in msgs:
            print(" -", m)
        raise SystemExit(1)  # non-zero exit means 'QC failed' (useful in automation)
    else:
        print("\nQC PASS ✅")

if __name__ == "__main__":
    main()
