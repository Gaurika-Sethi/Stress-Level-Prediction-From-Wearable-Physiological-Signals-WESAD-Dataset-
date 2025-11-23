import pandas as pd
from pathlib import Path

def merge_all_features():
    processed_dir = Path("data/processed")
    out_path = processed_dir / "all_features.csv"

    feature_files = sorted(processed_dir.glob("S*_features.csv"))

    if not feature_files:
        print("No feature files found in data/processed/.")
        return

    dfs = []
    for f in feature_files:
        df = pd.read_csv(f)
        subject = f.stem.split("_")[0]   
        df["subject"] = subject
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_path, index=False)

    print(f"[OK] Merged {len(feature_files)} subjects.")
    print(f"Saved → {out_path}   Shape: {all_df.shape}")

if __name__ == "__main__":
    merge_all_features()