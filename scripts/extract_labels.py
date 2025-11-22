import numpy as np
import pickle
from pathlib import Path

def extract_labels(subject):
    base = Path("data/raw/WESAD")
    pkl_path = base / subject / f"{subject}.pkl"

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    labels_700hz = data["label"]

    # wrist sampling rate = 64 Hz
    factor = 700 // 64  # ≈ 10
    labels_64hz = labels_700hz[::factor]

    out_dir = Path("data/interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{subject}_labels.npy", labels_64hz)

    print(f"[OK] Saved: {subject}_labels.npy (shape={labels_64hz.shape})")


if __name__ == "__main__":
    extract_labels("S10")