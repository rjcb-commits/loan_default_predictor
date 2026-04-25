"""Train a LightGBM model to predict loan default on the Lending Club dataset."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/loan.csv")
ARTIFACTS = Path("artifacts")
PLOTS = ARTIFACTS / "plots"
SAMPLE_FRACTION = 0.10
RANDOM_STATE = 42

FEATURES = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
]

CATEGORICAL = [
    "term",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]


def parse_term(s):
    if isinstance(s, str):
        return int(s.strip().split()[0])
    return np.nan


def parse_emp_length(s):
    if not isinstance(s, str):
        return np.nan
    s = s.strip()
    if "<" in s:
        return 0
    if "+" in s:
        return 10
    parts = s.split()
    return int(parts[0]) if parts and parts[0].isdigit() else np.nan


def parse_pct(s):
    if isinstance(s, str):
        s = s.replace("%", "").strip()
        return float(s) if s else np.nan
    return s


def load_and_clean() -> pd.DataFrame:
    print(f"Loading {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download Lending Club loan data from Kaggle "
            f"and place the main loan CSV at {DATA_PATH}."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)
    print(f"  {len(df):,} settled loans (default rate: {df['target'].mean():.1%})")

    if SAMPLE_FRACTION < 1.0:
        df, _ = train_test_split(
            df,
            train_size=SAMPLE_FRACTION,
            stratify=df["target"],
            random_state=RANDOM_STATE,
        )
        print(f"  Subsampled to {len(df):,} rows")

    df["term"] = df["term"].map(parse_term)
    df["int_rate"] = df["int_rate"].map(parse_pct)
    df["revol_util"] = df["revol_util"].map(parse_pct)
    df["emp_length"] = df["emp_length"].map(parse_emp_length)

    df = df[FEATURES + ["target"]].copy()
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")

    return df


def train_model(df: pd.DataFrame):
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )

    return model, X_train, X_test, y_train, y_test


def make_plots(model, X_test, y_test, proba, pred):
    PLOTS.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "roc.png", dpi=120)
    plt.close()

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.xticks([0, 1], ["Paid", "Charged off"])
    plt.yticks([0, 1], ["Paid", "Charged off"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(PLOTS / "confusion.png", dpi=120)
    plt.close()

    importance = (
        pd.DataFrame(
            {
                "feature": model.feature_name_,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=True)
        .tail(20)
    )
    plt.figure(figsize=(8, 7))
    plt.barh(importance["feature"], importance["importance"])
    plt.title("Top 20 features by importance")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(PLOTS / "importance.png", dpi=120)
    plt.close()


def save_metadata(model, X_train, y_test, proba, pred):
    metrics = {
        "auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "default_rate_test": float(y_test.mean()),
        "n_test": int(len(y_test)),
        "n_features": len(FEATURES),
    }

    feature_meta = {
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "categories": {
            col: [str(c) for c in X_train[col].cat.categories]
            for col in CATEGORICAL
        },
        "ranges": {
            col: {
                "min": float(np.nanmin(X_train[col])),
                "max": float(np.nanmax(X_train[col])),
            }
            for col in FEATURES
            if col not in CATEGORICAL
        },
    }

    sample_row = X_train.iloc[0].to_dict()
    sample = {}
    for k, v in sample_row.items():
        if isinstance(v, float) and np.isnan(v):
            sample[k] = None
        elif hasattr(v, "item"):
            sample[k] = v.item()
        else:
            sample[k] = str(v)

    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(ARTIFACTS / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)
    with open(ARTIFACTS / "sample.json", "w") as f:
        json.dump(sample, f, indent=2, default=str)

    return metrics


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    df = load_and_clean()
    model, X_train, X_test, y_train, y_test = train_model(df)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = save_metadata(model, X_train, y_test, proba, pred)
    make_plots(model, X_test, y_test, proba, pred)

    joblib.dump(model, ARTIFACTS / "model.pkl")

    print()
    print("=== Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved model and metadata to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
