from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_bbbp
from src.features import build_feature_table, scaffold_split, DESCRIPTOR_NAMES, FP_SIZE


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
(REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print(f"\n{name}")
    print(pd.Series(metrics).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, pred, target_names=["BBB-", "BBB+"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, pred))
    return metrics


def main():
    raw = load_bbbp()
    print("Raw shape:", raw.shape)
    print("\nClass distribution:")
    print(raw["p_np"].value_counts(dropna=False))

    raw = raw.dropna(subset=["smiles", "p_np"]).copy()
    raw = raw.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    clean, X, y = build_feature_table(raw, FP_SIZE)
    print(f"\nValid molecules after RDKit parsing: {len(clean)}")

    # Split using scaffold groups to reduce chemical similarity leakage.
    train_idx, test_idx = scaffold_split(clean, test_size=0.20, seed=42)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate(name, model, X_test, y_test))
        fitted[name] = model

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "bbbp_model.joblib")

    metadata = {
        "best_model": best_name,
        "feature_count": int(X.shape[1]),
        "descriptor_names": DESCRIPTOR_NAMES,
        "fingerprint_size": FP_SIZE,
        "split": "scaffold",
        "test_size": 0.20,
        "random_seed": 42,
        "classes": {"0": "BBB-", "1": "BBB+"},
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved best model: {best_name}")
    print(f"Model path: {MODEL_DIR / 'bbbp_model.joblib'}")


if __name__ == "__main__":
    main()
