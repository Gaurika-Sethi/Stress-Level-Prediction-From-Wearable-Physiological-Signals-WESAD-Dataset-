import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

def train_models():
    data_path = Path("data/processed/all_features.csv")
    if not data_path.exists():
        print("ERROR: all_features.csv not found. Run merge_all_features.py first.")
        return

    df = pd.read_csv(data_path)
    print("Loaded dataset:", df.shape)

    # ----- Filter labels -----
    # WESAD labels (0 = unknown, 1 = baseline, 2 = stress)
    # We keep only baseline vs stress
    df = df[df["label"].isin([1, 2])].copy()
    df["label"] = df["label"].map({1: 0, 2: 1})  # 0 = baseline, 1 = stress

    print("After label filter:", df.shape)

    # ----- Clean -----
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    print("After dropna:", df.shape)

    # ----- Prepare -----
    y = df["label"].values
    X = df.drop(columns=["label", "subject"], errors="ignore")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----- Scale -----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -------------------------
    # MODEL 1: RandomForest
    # -------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train_s, y_train)
    pred = rf.predict(X_test_s)
    proba = rf.predict_proba(X_test_s)[:, 1]

    print("\n=== RANDOM FOREST RESULTS ===")
    print(classification_report(y_test, pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))
    print("ROC AUC:", roc_auc_score(y_test, proba))

    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(rf, models_dir / "random_forest.pkl")
    joblib.dump(scaler, models_dir / "scaler_rf.pkl")
    print("[OK] Saved RandomForest + scaler")

    # -------------------------
    # MODEL 2: XGBoost
    # -------------------------
    xg = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    xg.fit(X_train, y_train)
    pred2 = xg.predict(X_test)
    proba2 = xg.predict_proba(X_test)[:, 1]

    print("\n=== XGBOOST RESULTS ===")
    print(classification_report(y_test, pred2))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred2))
    print("F1 Score:", f1_score(y_test, pred2))
    print("ROC AUC:", roc_auc_score(y_test, proba2))

    joblib.dump(xg, models_dir / "xgboost.pkl")
    print("[OK] Saved XGBoost model")

if __name__ == "__main__":
    train_models()