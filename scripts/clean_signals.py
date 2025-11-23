import pandas as pd
import numpy as np
from pathlib import Path

def clean_signals(subject):

    in_path = Path("data/interim") / f"{subject}_wrist.csv"
    df = pd.read_csv(in_path)

    df.loc[df["temp"] > 100, "temp"] = np.nan
    df["temp"] = df["temp"].interpolate(limit_direction='both')

    df.loc[df["eda"] > 10, "eda"] = np.nan
    df["eda"] = df["eda"].interpolate(limit_direction='both')

    df["bvp"] = df["bvp"].clip(-50, 200)

    FS = 64  
    df = df.iloc[10 * FS :]   

    out_path = Path("data/interim") / f"{subject}_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"[OK] {subject}: cleaned signals saved → {subject}_clean.csv")
    print("Shape:", df.shape)


if __name__ == "__main__":
    clean_signals("S10")