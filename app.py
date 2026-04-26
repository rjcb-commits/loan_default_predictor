"""Streamlit demo for the loan default predictor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
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


def format_value(v):
    if isinstance(v, float):
        if np.isnan(v):
            return "n/a"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.2f}"
    return str(v)


def sidebar_inputs(meta, sample):
    st.sidebar.header("Borrower features")
    inputs = {}
    for col in meta["features"]:
        label = col.replace("_", " ")
        if col in meta["categorical"]:
            options = meta["categories"][col]
            sample_val = sample.get(col)
            default_idx = options.index(sample_val) if sample_val in options else 0
            inputs[col] = st.sidebar.selectbox(label, options, index=default_idx)
        else:
            r = meta["ranges"][col]
            try:
                default = (
                    float(sample.get(col))
                    if sample.get(col) is not None
                    else (r["min"] + r["max"]) / 2
                )
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
        "LightGBM model trained on the Lending Club dataset. Move the sliders in "
        "the sidebar and the prediction updates live, alongside a per-feature "
        "contribution chart that shows exactly which inputs pushed this borrower "
        "above or below baseline default risk."
    )

    inputs = sidebar_inputs(meta, sample)

    X = pd.DataFrame([inputs])
    for col in meta["categorical"]:
        X[col] = pd.Categorical(X[col], categories=meta["categories"][col])

    prob = float(model.predict_proba(X)[0, 1])

    # Per-feature contributions (mathematically equivalent to SHAP for tree models).
    # Last column is the bias (model's expected log-odds output).
    contribs = model.predict(X, pred_contrib=True)
    feature_contribs = contribs[0, :-1]
    bias = float(contribs[0, -1])
    baseline_prob = 1.0 / (1.0 + np.exp(-bias))

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
        delta = prob - baseline_prob
        direction = "above" if delta >= 0 else "below"
        st.caption(
            f"Model baseline default rate: **{baseline_prob:.1%}**. "
            f"This borrower is **{abs(delta):.1%} {direction}** baseline."
        )

    with right:
        st.subheader("Why this prediction")
        st.caption(
            "Each bar shows how much a feature value pushed the prediction up "
            "(red, toward default) or down (green, toward repayment), in "
            "log-odds. Updates as you move the sliders."
        )
        contrib_df = pd.DataFrame(
            {
                "feature": model.feature_name_,
                "contribution": feature_contribs,
                "value": [format_value(X.iloc[0][f]) for f in model.feature_name_],
            }
        )
        contrib_df["abs"] = contrib_df["contribution"].abs()
        top = contrib_df.nlargest(10, "abs").sort_values("contribution")
        labels = [f"{row['feature']} = {row['value']}" for _, row in top.iterrows()]
        colors = ["#d62728" if c > 0 else "#2ca02c" for c in top["contribution"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(labels, top["contribution"], color=colors)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_xlabel("Contribution to log-odds of default")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Model details and diagnostics"):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Test AUC:** {metrics['auc']:.3f}")
            st.write(f"**Accuracy at 0.5 threshold:** {metrics['accuracy']:.3f}")
            st.write(f"**Default rate (test):** {metrics['default_rate_test']:.1%}")
            st.write(f"**Test set size:** {metrics['n_test']:,}")
            st.write(f"**Features used:** {metrics['n_features']}")
        with c2:
            roc_path = ARTIFACTS / "plots" / "roc.png"
            if roc_path.exists():
                st.image(str(roc_path), caption="ROC curve")

        cm_path = ARTIFACTS / "plots" / "confusion.png"
        imp_path = ARTIFACTS / "plots" / "importance.png"
        c3, c4 = st.columns(2)
        with c3:
            if cm_path.exists():
                st.image(str(cm_path), caption="Confusion matrix")
        with c4:
            if imp_path.exists():
                st.image(str(imp_path), caption="Global feature importance (gain)")


if __name__ == "__main__":
    main()
