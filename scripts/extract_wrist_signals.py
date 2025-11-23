import pandas as pd
import numpy as np
from pathlib import Path

def load_with_timestamps(csv_path, sample_rate):
    """
    Loads an Empatica wrist CSV:
    • Row 1 = start timestamp
    • Row 2 = sample rate
    • Remaining = data
    Returns a dataframe with a timestamp column.
    """
    with open(csv_path, "r") as f:
        lines = f.readlines()

    start_time = float(lines[0].strip())

    df = pd.read_csv(csv_path, skiprows=2, header=None)
    n = len(df)

    timestamps = start_time + np.arange(n) / sample_rate
    df["timestamp"] = timestamps

    return df


def extract_wrist_signals(subject):
    base = Path("data/raw/WESAD") / subject / f"{subject}_E4_Data"

    bvp_path  = base / "BVP.csv"
    eda_path  = base / "EDA.csv"
    temp_path = base / "TEMP.csv"

    bvp  = load_with_timestamps(bvp_path, 64.0)
    eda  = load_with_timestamps(eda_path, 4.0)
    temp = load_with_timestamps(temp_path, 4.0)

    bvp  = bvp.rename(columns={0: "bvp"})
    eda  = eda.rename(columns={0: "eda"})
    temp = temp.rename(columns={0: "temp"})

    merged = pd.merge_asof(
        bvp.sort_values("timestamp"),
        eda.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )

    merged = pd.merge_asof(
        merged,
        temp.sort_values("timestamp"),
        on="timestamp",
        direction="nearest"
    )

    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_dir / f"{subject}_wrist.csv", index=False)

    print(f"[OK] {subject}: wrist signals saved → {subject}_wrist.csv")
    print("Shape:", merged.shape)


if __name__ == "__main__":
    extract_wrist_signals("S10")