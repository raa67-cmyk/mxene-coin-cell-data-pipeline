# pipeline.py — end-to-end battery ETL for Arbin/Neware-style CSVs
from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd

# plotting (saves PNGs; no GUI window)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Normalization (Arbin-first)
# ---------------------------
def normalize(csv_path: Path, out_parquet: Path) -> pd.DataFrame:
    """Read vendor CSV, map headers to a canonical schema, fix units/signs, and save Parquet."""
    # 1) Read raw CSV first (no parse_dates here)
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        # fall back to semicolon-delimited (EU exports)
        df = pd.read_csv(csv_path, low_memory=False, sep=";")

    # trim column name whitespace
    df.columns = [c.strip() for c in df.columns]

    # 2) Header maps (Arbin + common Neware variants)
    arbin_map = {
        "Date_Time": "timestamp",
        "Date Time": "timestamp",
        "Cycle_Index": "cycle_index",
        "Step_Index": "step_index",
        "Step_Name": "step_type",
        "Current(A)": "current_a",
        "Current(mA)": "current_mA",
        "Voltage(V)": "voltage_v",
        "Voltage(mV)": "voltage_mV",
        "Temperature(C)": "temp_c",
        "Charge_Capacity(Ah)": "charge_ah",
        "Charge_Capacity(mAh)": "charge_mAh",
        "Discharge_Capacity(Ah)": "discharge_ah",
        "Discharge_Capacity(mAh)": "discharge_mAh",
        "Test Time (s)": "test_time_s",
        "Test_Time(s)": "test_time_s",
    }
    neware_map = {
        "Record Time": "timestamp",
        "Cycle": "cycle_index",
        "Step": "step_index",
        "Mode": "step_type",
        "Status": "step_type",
        "Current(A)": "current_a",
        "Current(mA)": "current_mA",
        "Voltage(V)": "voltage_v",
        "Voltage(mV)": "voltage_mV",
        "NTC": "temp_c",
        "Temperature(℃)": "temp_c",
        "CapCharge(Ah)": "charge_ah",
        "CapDischarge(Ah)": "discharge_ah",
        "CHARGE_Ah": "charge_ah",
        "DISCHARGE_Ah": "discharge_ah",
        "Capacity Charge(mAh)": "charge_mAh",
        "Capacity Discharge(mAh)": "discharge_mAh",
        "Time(s)": "test_time_s",
        "Test Time(s)": "test_time_s",
    }

    def soft_rename(frame: pd.DataFrame, mapping: dict) -> pd.DataFrame:
        exists = {k: v for k, v in mapping.items() if k in frame.columns}
        return frame.rename(columns=exists)

    # apply maps (Arbin first, then Neware)
    df = soft_rename(df, arbin_map)
    df = soft_rename(df, neware_map)

    # 3) Build canonical columns
    # timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "test_time_s" in df.columns:
        t0 = pd.Timestamp("1970-01-01")
        df["timestamp"] = t0 + pd.to_timedelta(pd.to_numeric(df["test_time_s"], errors="coerce"), unit="s")
    else:
        raise ValueError("No timestamp-like column found (expected Date_Time/Record Time or Test Time (s)).")

    # cycle & step indices
    if "cycle_index" not in df.columns:
        for cand in ["Cycle_Index", "Cycle", "cycle", "CycleIndex"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "cycle_index"})
                break
    if "step_index" not in df.columns:
        for cand in ["Step_Index", "Step", "StepIndex", "Index"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "step_index"})
                break
    df["cycle_index"] = pd.to_numeric(df.get("cycle_index"), errors="coerce").astype("Int64")
    df["step_index"] = pd.to_numeric(df.get("step_index"), errors="coerce").astype("Int64")

    # step type
    if "step_type" not in df.columns:
        # fallback guess based on current sign
        cur_col = "current_a" if "current_a" in df.columns else ("current_mA" if "current_mA" in df.columns else None)
        if cur_col:
            cur = pd.to_numeric(df[cur_col], errors="coerce")
            df["step_type"] = np.where(cur < 0, "CC_DIS", np.where(cur > 0, "CC_CHG", "REST"))
        else:
            df["step_type"] = "REST"
    df["step_type"] = df["step_type"].astype(str).str.upper().replace({
        "CC CHARGE": "CC_CHG", "CCC": "CC_CHG", "CHG": "CC_CHG",
        "CV CHARGE": "CV", "CV": "CV",
        "CC DISCHARGE": "CC_DIS", "CCD": "CC_DIS", "DCHG": "CC_DIS",
        "REST": "REST", "PAUSE": "REST", "IDLE": "REST",
    })

    # units → A/Ah/V
    if "current_a" not in df.columns and "current_mA" in df.columns:
        df["current_a"] = pd.to_numeric(df["current_mA"], errors="coerce") / 1000.0
    if "voltage_v" not in df.columns and "voltage_mV" in df.columns:
        df["voltage_v"] = pd.to_numeric(df["voltage_mV"], errors="coerce") / 1000.0
    if "charge_ah" not in df.columns and "charge_mAh" in df.columns:
        df["charge_ah"] = pd.to_numeric(df["charge_mAh"], errors="coerce") / 1000.0
    if "discharge_ah" not in df.columns and "discharge_mAh" in df.columns:
        df["discharge_ah"] = pd.to_numeric(df["discharge_mAh"], errors="coerce") / 1000.0

    # temperature optional
    if "temp_c" in df.columns:
        df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")

    # discharge-negative convention (flip if DIS mostly positive)
    dis_mask = df["step_type"].str.contains("DIS", na=False)
    if dis_mask.any():
        if (pd.to_numeric(df.loc[dis_mask, "current_a"], errors="coerce") > 0).mean() > 0.8:
            df["current_a"] = -pd.to_numeric(df["current_a"], errors="coerce")

    # 4) Keep canonical columns
    needed = ["timestamp", "cycle_index", "step_index", "step_type",
              "current_a", "voltage_v", "temp_c", "charge_ah", "discharge_ah"]
    for c in needed:
        if c not in df.columns:
            if c == "temp_c":
                df["temp_c"] = np.nan
            else:
                raise ValueError(f"Missing required column after normalization: {c}")

    # 5) Sort & write Parquet
    df = df.sort_values("timestamp").reset_index(drop=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df[needed].to_parquet(out_parquet, index=False)
    return df[needed]


# ---------------------------
# Feature engineering
# ---------------------------
def capacity_ce_per_cycle(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby("cycle_index", sort=True):
        qdis = pd.to_numeric(g["discharge_ah"], errors="coerce").dropna().iloc[-1]
        qchg = pd.to_numeric(g["charge_ah"], errors="coerce").dropna().iloc[-1]
        ce = np.nan if (pd.isna(qchg) or qchg == 0) else (qdis / qchg)
        rows.append({"cycle_index": int(k), "Q_dis_Ah": qdis, "Q_chg_Ah": qchg, "CE": ce})
    feat = pd.DataFrame(rows).sort_values("cycle_index")
    feat["q_norm"] = feat["Q_dis_Ah"] / feat["Q_dis_Ah"].iloc[0]
    return feat


def energy_wh_per_cycle(df: pd.DataFrame) -> pd.DataFrame:
    def e_wh(g):
        dis = g[g["step_type"].astype(str).str.contains("DIS")]
        if dis.shape[0] < 2:
            return np.nan
        # convert timestamps to seconds (float)
        t = dis["timestamp"].astype("int64").to_numpy() / 1e9
        p = (dis["voltage_v"] * dis["current_a"]).to_numpy()
        e_ws = np.trapz(p, t)  # watt-seconds
        return abs(e_ws) / 3600.0

    rows = [{"cycle_index": int(k), "E_dis_Wh": e_wh(g)} for k, g in df.groupby("cycle_index", sort=True)]
    return pd.DataFrame(rows)


def ir_c2_per_cycle(df: pd.DataFrame, rated_ah: float) -> pd.DataFrame:
    target = 0.5 * rated_ah  # C/2
    def ir_one(g):
        dis = g[g["step_type"].astype(str).str.contains("DIS")]
        if dis.empty:
            return np.nan
        idx = (dis["current_a"].abs() - target).abs().idxmin()
        w = 1  # tiny window for toy/sample data; widen for real data
        pre = dis.loc[max(dis.index.min(), idx - w): idx - 1]
        post = dis.loc[idx: min(idx + w, dis.index.max())]
        if pre.empty or post.empty:
            return np.nan
        dV = post["voltage_v"].median() - pre["voltage_v"].median()
        dI = post["current_a"].median() - pre["current_a"].median()
        if dI == 0 or pd.isna(dI):
            return np.nan
        return abs(dV / dI)

    rows = [{"cycle_index": int(k), "IR_C2_ohm": ir_one(g)} for k, g in df.groupby("cycle_index", sort=True)]
    return pd.DataFrame(rows)


def dqdv_peak_per_cycle(df: pd.DataFrame, dV: float = 0.05) -> pd.DataFrame:
    def vpeak(g):
        dis = g[g["step_type"].astype(str).str.contains("DIS")]
        if dis.shape[0] < 3:
            return np.nan
        V = dis["voltage_v"].to_numpy()
        Q = (dis["discharge_ah"] - dis["discharge_ah"].min()).to_numpy()
        order = np.argsort(V); V, Q = V[order], Q[order]
        if V[-1] - V[0] < dV:
            return np.nan
        vgrid = np.arange(V[0], V[-1], dV)
        qgrid = np.interp(vgrid, V, Q)
        dqdv = np.gradient(qgrid, dV)
        return float(vgrid[int(np.argmax(dqdv))])

    rows, vref = [], None
    for k, g in df.groupby("cycle_index", sort=True):
        vpk = vpeak(g)
        if vref is None and not pd.isna(vpk):
            vref = vpk
        shift = (vpk - vref) * 1000.0 if (vref is not None and not pd.isna(vpk)) else np.nan
        rows.append({"cycle_index": int(k), "dQdV_peak_V": vpk, "dQdV_shift_mV": shift})
    return pd.DataFrame(rows)


def fade_and_rul(feat: pd.DataFrame, eol=0.80):
    x = feat["cycle_index"].to_numpy(dtype=float)
    y = feat["q_norm"].to_numpy(dtype=float)
    if len(x) < 2:
        return {"fade_slope_pct_per_cycle": np.nan, "cycles_to_80pct": np.nan}
    m, b = np.polyfit(x, y, 1)
    slope_pct = m * 100.0
    n_eol = (eol - b) / m if m != 0 else math.nan
    return {
        "fade_slope_pct_per_cycle": float(slope_pct),
        "cycles_to_80pct": float(n_eol) if not math.isnan(n_eol) else np.nan,
    }


def quick_plots(feat: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(feat["cycle_index"], feat["Q_dis_Ah"], marker="o")
    plt.xlabel("Cycle"); plt.ylabel("Discharge Capacity (Ah)")
    plt.title("Capacity vs Cycle"); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / "plot_capacity.png")

    plt.figure()
    plt.plot(feat["cycle_index"], feat["CE"], marker="o")
    plt.xlabel("Cycle"); plt.ylabel("Coulombic Efficiency")
    plt.title("CE vs Cycle"); plt.grid(True); plt.tight_layout()
    plt.savefig(out_dir / "plot_ce.png")


# ---------------------------
# Main
# ---------------------------
def main():
    ROOT = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Simple battery cycle-life ETL (Arbin/Neware CSV).")
    parser.add_argument("--in", dest="in_file", required=True, help="Path to input CSV")
    parser.add_argument("--cell", dest="cell_id", default="CELL01", help="Cell identifier")
    parser.add_argument("--rated_ah", type=float, default=3.0, help="Rated capacity (Ah) for C-rate calcs")
    args = parser.parse_args()

    raw = Path(args.in_file)
    interim = ROOT / "data" / "interim"
    processed = ROOT / "data" / "processed"
    interim.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    parquet_path = interim / f"{args.cell_id}_timeseries.parquet"

    print("[1/6] Normalizing & saving Parquet …")
    df = normalize(raw, parquet_path)

    print("[2/6] Computing capacity & CE …")
    cap = capacity_ce_per_cycle(df)

    print("[3/6] Computing energy per cycle …")
    ener = energy_wh_per_cycle(df)

    print("[4/6] Computing IR @ C/2 …")
    ir = ir_c2_per_cycle(df, args.rated_ah)

    print("[5/6] Computing dQ/dV peak & shift …")
    dqdv = dqdv_peak_per_cycle(df, dV=0.05)  # use 0.005 for real data resolution

    feat = cap.merge(ener, on="cycle_index", how="left") \
              .merge(ir,   on="cycle_index", how="left") \
              .merge(dqdv, on="cycle_index", how="left")

    full_csv = processed / f"{args.cell_id}_features_full.csv"
    feat.to_csv(full_csv, index=False)
    print("→ wrote", full_csv)

    stats = fade_and_rul(feat, eol=0.80)
    summary = pd.DataFrame([{
        "cell_id": args.cell_id,
        "Q0_Ah": float(feat["Q_dis_Ah"].iloc[0]),
        **stats
    }])
    summary_csv = processed / f"{args.cell_id}_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("→ wrote", summary_csv)

    print("[6/6] Saving quick plots …")
    quick_plots(feat, processed)
    print("done.")


if __name__ == "__main__":
    main()
