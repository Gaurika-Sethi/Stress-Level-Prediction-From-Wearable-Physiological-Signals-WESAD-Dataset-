import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

FEATURES_PATH = Path("data/processed/all_features.csv")

def load_data():
    df = pd.read_csv(FEATURES_PATH)

    df = df[df["label"].isin([1,2])].copy()
    df["label"] = df["label"].map({1:0, 2:1})

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def loso_training():
    df = load_data()

    subjects = df["subject"].unique()
    subjects = sorted(subjects, key=lambda x: int(x[1:]))  

    print("Subjects:", subjects)

    results = []

    for test_subj in subjects:
        print(f"\n============================")
        print(f" Testing on {test_subj} (LOSO)")
        print(f"============================")

        train_df = df[df["subject"] != test_subj]
        test_df  = df[df["subject"] == test_subj]

        X_train = train_df.drop(columns=["label", "subject"])
        y_train = train_df["label"]

        X_test  = test_df.drop(columns=["label", "subject"])
        y_test  = test_df["label"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        )
        model.fit(X_train_s, y_train)

        pred = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:,1]

        acc = accuracy_score(y_test, pred)
        f1  = f1_score(y_test, pred)
        try:
            auc = roc_auc_score(y_test, proba)
        except:
            auc = np.nan

        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        results.append([test_subj, acc, f1, auc])

    res_df = pd.DataFrame(results, columns=["subject", "accuracy", "f1", "auc"])
    out_path = Path("models/loso_results.csv")
    res_df.to_csv(out_path, index=False)

    print("\n============================")
    print("LOSO RESULTS SAVED → models/loso_results.csv")
    print("============================")
    print(res_df)
    print("\nMean Accuracy:", res_df["accuracy"].mean())
    print("Mean F1:", res_df["f1"].mean())
    print("Mean AUC:", res_df["auc"].mean())


if __name__ == "__main__":
    loso_training()