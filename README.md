# Stress Detection from Wearable Physiological Signals

![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Model-Scikit--Learn-yellow)

## Overview
This project builds a research-grade stress detection model using wearable physiological signals (BVP, EDA, Temperature) from the WESAD dataset.
It includes a complete end-to-end biomedical signal processing pipeline, suitable for:

- DRDO labs (DIPAS, INMAS, DEBEL)
- IIT/IIIT research internships
- Wearable device R&D
- Human stress monitoring prototypes

The pipeline applies signal filtering → windowing → feature extraction → LOSO evaluation → model explainability to ensure realistic, person-independent performance.

## Dataset & Preprocessing 

### Dataset
This project uses the WESAD (Wearable Stress and Affect Detection) dataset — a multimodal physiological dataset collected from chest and wrist sensors.

Physiological channels used:

- BVP (Blood Volume Pulse)
- EDA (Electrodermal Activity)
- TEMP (Peripheral Temperature)

Data Characteristics:

- 15 subjects
- 3 conditions (Baseline, Stress, Amusement)

Wrist sensor sampling:

- BVP → 64 Hz
- EDA → 4 Hz
- Temperature → 4 Hz

Only baseline vs stress labels were used for binary classification.

### Preprocessing Pipeline

**Raw Signal Extraction**
All wrist sensor files were extracted from subject folders and aligned to create a synchronized data stream.

**Signal Alignment**
BVP, EDA, TEMP were aligned using timestamps and re-sampled for consistency.

**Filtering**
Applied physiologically appropriate filters:

- BVP: 0.5–4 Hz bandpass
- EDA: low-pass (<0.6 Hz)
- Temp: low-pass (<0.1 Hz)

This removes noise, motion artifacts, and sensor spikes.

**Signal Alignment**
Signals were segmented into 5-second non-overlapping windows, chosen because:

- EDA responses take ~4–8s
- BVP HRV-like metrics stabilize at ~5s
- Temperature changes are slow but detectable

## Feature Engineering

A total of 17 research-grade features were extracted:

**HRV-like BVP Features**

- Heart Rate from peaks
- RMSSD
- SDNN
- pNN50
- Pulse peak-to-peak amplitude
- Pulse signal energy
- Pulse Rate Variability (PRV-SDNN)

**EDA Features**

- Mean EDA
- EDA standard deviation
- EDA slope
- SCR count
- SCR peak amplitude
- SCR rise rate

**Temperature Features**

- Temperature mean
- Temperature slope
- Temperature derivative

These features capture autonomic nervous system activity responsible for stress response.

### Machine Learning & Evaluation

**Models Used**

- Random Forest
- XGBoost (optional)

**Evaluation Strategy: LOSO (Leave-One-Subject-Out)**

This is the gold standard for wearable physiology:

*Train on N–1 subjects, test on the remaining subject.*

This ensures true person-independent generalization — mandatory for biomedical research.

**Metrics Reported**

- Accuracy
- F1 Score
- ROC AUC
- Per-subject performance
- Confusion matrices

Typical performance:

- Accuracy: ~0.92–0.97
- AUC: ~0.95–0.99

### Visualizations

The notebook includes:

**Raw vs Filtered Signal Plots**

BVP, EDA, TEMP before/after filtering.

**PCA Projection**

Shows clear separation between baseline vs stress.

**Feature Importance**

- Random Forest importance
- SHAP permutation-based summary

**LOSO Performance Plots**

- Accuracy per subject
- F1 per subject
- AUC per subject

### Explainability (Optional SHAP Analysis)

A lightweight SHAP version (≤30 seconds) is included:

- Permutation-based SHAP
- Global summary plot
- Bar-plot feature importance

This provides insight into why the model predicts stress, supporting scientific interpretability.

### Results Summary

The model demonstrates:

- Strong generalization across subjects
- Clear physiologically interpretable patterns
- Robust feature significance (EDA + HRV dominate)

## Limitations
- WESAD contains only 15 subjects
- Wrist BVP is noisier than ECG
- Label boundaries are coarse
- LOSO evaluation is expensive
- Some signals may contain motion artifacts

## Future Work
- Deep learning (1D CNN / LSTM on raw BVP/EDA)
- Fusion of wrist + chest sensors
- Real-time stress detection system
- Deployment on Jetson Nano
- Integrating accelerometer (ACC) features
- Better SCR decomposition with cvxEDA

## Installation

Follow these steps to run the project locally:

### Clone the repository  
```bash
   git clone https://github.com/Gaurika-Sethi/Stress-Level-Prediction-From-Wearable-Physiological-Signals-WESAD-Dataset-.git
  ```
### Create a virtual environment 
```bash
python -m venv venv
```
### Activate the environment 

Windows:
```bash
venv\Scripts\activate
```
macOS/Linux:
```bash
source venv/bin/activate
```

### Install dependencies 
```bash
pip install -r requirements.txt
```
### Run the Streamlit app 
```bash
streamlit run app.py
```
## Usage

Follow these steps to run the project locally:

### jupyter notebook stress_detection.ipynb
```bash
 jupyter notebook stress_detection.ipynb
```
### Folder Structure

```bash
data/
│── raw/
│── interim/
│── processed/
models/
notebooks/
scripts/
```

## Contact
- LinkedIn: https://www.linkedin.com/in/gaurika-sethi-53043b321
- Medium: https://medium.com/@pixelsnsyntax
- Twitter: https://twitter.com/pixelsnsyntax

Project Link: [Repository link](https://github.com/Gaurika-Sethi/Stress-Level-Prediction-From-Wearable-Physiological-Signals-WESAD-Dataset-.git)

## License

This project is licensed under the **MIT License**  see the [LICENSE](LICENSE) file for full details. 
