# step13_collate.py — collate without using DataFrame.insert()
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

def add_cell_id(df: pd.DataFrame, cid: str) -> pd.DataFrame:
    # add if missing
    if "cell_id" not in df.columns:
        df["cell_id"] = cid
    # if present but empty, fill
    elif df["cell_id"].isna().all() or (df["cell_id"].astype(str).str.len() == 0).all():
        df["cell_id"] = cid
    # put cell_id first
    cols = ["cell_id"] + [c for c in df.columns if c != "cell_id"]
    return df[cols]

# ----- per-cycle features -----
feat_files = sorted(PROC.glob("*_features_full.csv"))
feat_frames = []
for f in feat_files:
    cid = f.stem[:-len("_features_full")] if f.stem.endswith("_features_full") else f.stem
    df = pd.read_csv(f)
    df = add_cell_id(df, cid)
    feat_frames.append(df)

if feat_frames:
    out_feat = PROC / "_all_features.csv"
    pd.concat(feat_frames, ignore_index=True).to_csv(out_feat, index=False)
    print("Wrote:", out_feat)
else:
    print("No *_features_full.csv found in", PROC)

# ----- per-cell summaries -----
sum_files = sorted(PROC.glob("*_summary.csv"))
sum_frames = []
for f in sum_files:
    cid = f.stem[:-len("_summary")] if f.stem.endswith("_summary") else f.stem
    df = pd.read_csv(f)
    df = add_cell_id(df, cid)
    sum_frames.append(df)

if sum_frames:
    out_sum = PROC / "_all_summaries.csv"
    pd.concat(sum_frames, ignore_index=True).to_csv(out_sum, index=False)
    print("Wrote:", out_sum)
else:
    print("No *_summary.csv found in", PROC)

