from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
parquet_path = ROOT / "data" / "interim" / "cell01_timeseries.parquet"
out_csv = ROOT / "data" / "processed" / "cell01_dqdv_step9.csv"

df = pd.read_parquet(parquet_path)

def dqdv_peak_for_cycle(g: pd.DataFrame, dV: float = 0.05):
    """Return (v_peak, ok) where v_peak is the voltage of max dQ/dV on discharge."""
    dis = g[g["step_type"].astype(str).str.contains("DIS")]
    if dis.shape[0] < 3:
        return np.nan, False
    V = dis["voltage_v"].to_numpy()
    Q = (dis["discharge_ah"] - dis["discharge_ah"].min()).to_numpy()
    # enforce increasing voltage for interpolation stability
    order = np.argsort(V)
    V, Q = V[order], Q[order]
    if V[-1] - V[0] < dV:  # too narrow to meaningfully grid
        return np.nan, False
    vgrid = np.arange(V[0], V[-1], dV)
    qgrid = np.interp(vgrid, V, Q)
    dqdv = np.gradient(qgrid, dV)
    idx = int(np.argmax(dqdv))
    return float(vgrid[idx]), True

rows, v_ref = [], None
for k, g in df.groupby("cycle_index", sort=True):
    vpk, ok = dqdv_peak_for_cycle(g, dV=0.05)  # coarse 50 mV grid for toy data
    if v_ref is None and ok:
        v_ref = vpk
    shift_mv = (vpk - v_ref) * 1000.0 if (v_ref is not None and ok) else np.nan
    rows.append({"cycle_index": int(k), "dQdV_peak_V": vpk, "dQdV_shift_mV": shift_mv})

out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print(pd.read_csv(out_csv))
