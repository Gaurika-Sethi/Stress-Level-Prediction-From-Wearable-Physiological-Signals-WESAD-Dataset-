import pandas as pd
import numpy as np
from pathlib import Path

def clean_signals(subject):

    # Load wrist data
    in_path = Path("data/interim") / f"{subject}_wrist.csv"
    df = pd.read_csv(in_path)

    # -------------------------
    # CLEAN TEMP SENSOR
    # -------------------------
    # Remove impossible TEMP values (device warmup can create 350+ values)
    df.loc[df["temp"] > 100, "temp"] = np.nan
    df["temp"] = df["temp"].interpolate(limit_direction='both')

    # -------------------------
    # CLEAN EDA SENSOR
    # -------------------------
    # Remove abnormal spikes
    df.loc[df["eda"] > 10, "eda"] = np.nan
    df["eda"] = df["eda"].interpolate(limit_direction='both')

    # -------------------------
    # CLEAN BVP SIGNAL
    # -------------------------
    # Clip to reasonable biological ranges
    df["bvp"] = df["bvp"].clip(-50, 200)

    # -------------------------
    # REMOVE FIRST 10 SECONDS (warm-up noise)
    # -------------------------
    FS = 64  # BVP sampling rate
    df = df.iloc[10 * FS :]   # removes 640 samples

    # Save cleaned output
    out_path = Path("data/interim") / f"{subject}_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"[OK] {subject}: cleaned signals saved → {subject}_clean.csv")
    print("Shape:", df.shape)


if __name__ == "__main__":
    clean_signals("S10")