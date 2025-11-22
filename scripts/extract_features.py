import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks

# ==========================
# HRV-LIKE FEATURE FUNCTIONS
# ==========================

def compute_hrv_features(bvp_window):
    """Basic HRV-like features from BVP differences."""
    diffs = np.diff(bvp_window)

    rmssd = np.sqrt(np.mean(diffs ** 2)) if len(diffs) > 1 else np.nan
    sdnn = np.std(diffs) if len(diffs) > 1 else np.nan
    pnn50 = np.mean(np.abs(diffs) > 0.05) if len(diffs) > 1 else np.nan

    # Simple peak detection for HR (not PRV)
    peaks = np.where(
        (bvp_window[1:-1] > bvp_window[:-2]) &
        (bvp_window[1:-1] > bvp_window[2:])
    )[0]

    hr = (len(peaks) * 20) if len(peaks) > 0 else np.nan  

    return hr, rmssd, sdnn, pnn50


# ==============================
# ADVANCED FEATURE EXTRACTION
# ==============================

def compute_advanced_bvp_features(bvp_window, fs=64):
    """Extract BVP peaks, amplitude, energy, PRV."""
    # Proper peak detection
    peaks, _ = find_peaks(bvp_window, distance=int(0.5 * fs))

    # Peak-to-peak amplitude
    if len(peaks) > 1:
        bvp_pp_amp = np.mean(np.diff(bvp_window[peaks]))
    else:
        bvp_pp_amp = 0

    # Energy
    bvp_energy = np.sum(bvp_window ** 2)

    # PRV (Pulse Rate Variability)
    if len(peaks) > 1:
        ibi = np.diff(peaks) / fs
        prv_sdnn = np.std(ibi)
    else:
        prv_sdnn = 0

    return bvp_pp_amp, bvp_energy, prv_sdnn


def compute_advanced_eda_features(eda_window, fs=4):
    """SCR count, amplitude, rise rate."""
    scr_peaks, _ = find_peaks(eda_window, distance=fs)  # 1 second apart
    scr_count = len(scr_peaks)

    if scr_count > 0:
        scr_mean_amp = np.mean(eda_window[scr_peaks])
    else:
        scr_mean_amp = 0

    if scr_count > 1:
        scr_rise_rate = np.max(np.diff(eda_window[scr_peaks]))
    else:
        scr_rise_rate = 0

    return scr_count, scr_mean_amp, scr_rise_rate


def compute_temp_features(temp_window):
    """Temp variability and derivative."""
    temp_var = np.std(temp_window)
    temp_change = np.mean(np.diff(temp_window)) if len(temp_window) > 1 else 0
    return temp_var, temp_change


# ==============================
# MAIN EXTRACTION FUNCTION
# ==============================

def extract_features(subject):

    clean_path = Path("data/interim") / f"{subject}_clean.csv"
    labels_path = Path("data/interim") / f"{subject}_labels.npy"

    df = pd.read_csv(clean_path)
    labels = np.load(labels_path)

    min_len = min(len(df), len(labels))
    df = df.iloc[:min_len]
    df["label"] = labels[:min_len]

    FS = 64
    window_size = 3 * FS   # 3-second windows (modify if needed)
    step = FS              # 1-second stride

    rows = []

    for start in range(0, len(df) - window_size, step):
        end = start + window_size
        window = df.iloc[start:end]

        bvp = window["bvp"].values
        eda = window["eda"].values
        temp = window["temp"].values

        # ----- Basic HRV features -----
        hr, rmssd, sdnn, pnn50 = compute_hrv_features(bvp)

        # ----- Advanced BVP features -----
        bvp_pp_amp, bvp_energy, prv_sdnn = compute_advanced_bvp_features(bvp)

        # ----- Basic EDA features -----
        eda_mean = eda.mean()
        eda_std = eda.std()
        eda_slope = (eda[-1] - eda[0]) / window_size

        # ----- Advanced EDA features -----
        scr_count, scr_mean_amp, scr_rise_rate = compute_advanced_eda_features(eda, fs=4)

        # ----- Temperature features -----
        temp_mean = temp.mean()
        temp_slope = (temp[-1] - temp[0]) / window_size
        temp_var, temp_change = compute_temp_features(temp)

        # Label for the window
        label = window["label"].mode()[0]

        rows.append([
            hr, rmssd, sdnn, pnn50,
            eda_mean, eda_std, eda_slope,
            temp_mean, temp_slope,
            bvp_pp_amp, bvp_energy, prv_sdnn,
            scr_count, scr_mean_amp, scr_rise_rate,
            temp_var, temp_change,
            label
        ])

    # Create dataframe
    feature_cols = [
        "hr", "rmssd", "sdnn", "pnn50",
        "eda_mean", "eda_std", "eda_slope",
        "temp_mean", "temp_slope",
        "bvp_pp_amp", "bvp_energy", "prv_sdnn",
        "scr_count", "scr_mean_amp", "scr_rise_rate",
        "temp_var", "temp_change",
        "label"
    ]

    features = pd.DataFrame(rows, columns=feature_cols)

    # Save
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{subject}_features.csv"
    features.to_csv(out_path, index=False)

    print(f"[OK] {subject}: features saved → {out_path.name}")
    print("Shape:", features.shape)


if __name__ == "__main__":
    extract_features("S10")