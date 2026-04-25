"""Streamlit demo for the loan default predictor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ARTIFACTS = Path("artifacts")

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "model.pkl")


@st.cache_data
def load_json(name: str):
    with open(ARTIFACTS / name) as f:
        return json.load(f)


def main():
    if not (ARTIFACTS / "model.pkl").exists():
        st.error(
            "No trained model found. Run `python train.py` first to generate "
            "`artifacts/model.pkl`."
        )
        return

    model = load_model()
    meta = load_json("feature_meta.json")
    metrics = load_json("metrics.json")
    sample = load_json("sample.json")

    st.title("Loan Default Predictor")
    st.write(
        "LightGBM model trained on the Lending Club dataset to estimate the "
        "probability that a loan will be charged off rather than fully repaid. "
        "Adjust the borrower features in the sidebar and the prediction updates "
        "in real time."
    )

    inputs = sidebar_inputs(meta, sample)

    X = pd.DataFrame([inputs])
    for col in meta["categorical"]:
        X[col] = pd.Categorical(X[col], categories=meta["categories"][col])

    prob = float(model.predict_proba(X)[0, 1])

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Prediction")
        st.metric("Probability of default", f"{prob:.1%}")
        if prob < 0.10:
            st.success("Low risk")
        elif prob < 0.25:
            st.info("Moderate risk")
        else:
            st.warning("Elevated risk")

    with right:
        st.subheader("Top features by gain")
        importance = (
            pd.DataFrame(
                {
                    "feature": model.feature_name_,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=True)
            .tail(10)
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(importance["feature"], importance["importance"])
        ax.set_xlabel("Importance (gain)")
        fig.tight_layout()
        st.pyplot(fig)

    with st.expander("Model details"):
        st.write(f"**Test AUC:** {metrics['auc']:.3f}")
        st.write(f"**Accuracy at 0.5 threshold:** {metrics['accuracy']:.3f}")
        st.write(f"**Default rate in test set:** {metrics['default_rate_test']:.1%}")
        st.write(f"**Test set size:** {metrics['n_test']:,}")
        st.write(f"**Features used:** {metrics['n_features']}")


def sidebar_inputs(meta, sample):
    st.sidebar.header("Borrower features")
    inputs = {}
    for col in meta["features"]:
        label = col.replace("_", " ")
        if col in meta["categorical"]:
            options = meta["categories"][col]
            sample_val = sample.get(col)
            default_idx = (
                options.index(sample_val) if sample_val in options else 0
            )
            inputs[col] = st.sidebar.selectbox(label, options, index=default_idx)
        else:
            r = meta["ranges"][col]
            try:
                default = float(sample.get(col)) if sample.get(col) is not None else (r["min"] + r["max"]) / 2
            except (TypeError, ValueError):
                default = (r["min"] + r["max"]) / 2
            default = max(r["min"], min(default, r["max"]))
            step = (r["max"] - r["min"]) / 100 if r["max"] > r["min"] else 1.0
            inputs[col] = st.sidebar.slider(
                label,
                min_value=float(r["min"]),
                max_value=float(r["max"]),
                value=float(default),
                step=float(step),
            )
    return inputs


if __name__ == "__main__":
    main()
