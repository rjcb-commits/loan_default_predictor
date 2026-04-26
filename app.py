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


# Hand-tuned borrower profiles. All values are clamped to valid options/ranges
# at runtime via safe_preset(), so if the trained model's category lists or
# ranges differ slightly from these guesses the app still works.
PRESETS = {
    "low_risk": {
        "loan_amnt": 7000.0,
        "term": "36",
        "int_rate": 7.5,
        "installment": 220.0,
        "grade": "A",
        "emp_length": 10.0,
        "home_ownership": "MORTGAGE",
        "annual_inc": 95000.0,
        "verification_status": "Source Verified",
        "purpose": "credit_card",
        "addr_state": "CA",
        "dti": 9.0,
        "delinq_2yrs": 0.0,
        "fico_range_low": 770.0,
        "inq_last_6mths": 0.0,
        "open_acc": 14.0,
        "pub_rec": 0.0,
        "revol_bal": 5000.0,
        "revol_util": 15.0,
        "total_acc": 30.0,
    },
    "typical": {
        "loan_amnt": 12000.0,
        "term": "36",
        "int_rate": 13.0,
        "installment": 400.0,
        "grade": "C",
        "emp_length": 5.0,
        "home_ownership": "MORTGAGE",
        "annual_inc": 65000.0,
        "verification_status": "Source Verified",
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "dti": 18.0,
        "delinq_2yrs": 0.0,
        "fico_range_low": 690.0,
        "inq_last_6mths": 0.0,
        "open_acc": 11.0,
        "pub_rec": 0.0,
        "revol_bal": 12000.0,
        "revol_util": 50.0,
        "total_acc": 24.0,
    },
    "high_risk": {
        "loan_amnt": 28000.0,
        "term": "60",
        "int_rate": 22.0,
        "installment": 770.0,
        "grade": "E",
        "emp_length": 1.0,
        "home_ownership": "RENT",
        "annual_inc": 38000.0,
        "verification_status": "Not Verified",
        "purpose": "debt_consolidation",
        "addr_state": "FL",
        "dti": 32.0,
        "delinq_2yrs": 2.0,
        "fico_range_low": 645.0,
        "inq_last_6mths": 3.0,
        "open_acc": 8.0,
        "pub_rec": 1.0,
        "revol_bal": 18000.0,
        "revol_util": 88.0,
        "total_acc": 14.0,
    },
}


@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "model.pkl")


@st.cache_data
def load_json(name: str):
    with open(ARTIFACTS / name) as f:
        return json.load(f)


def format_value(v):
    if isinstance(v, float) and pd.isna(v):
        return "n/a"
    if isinstance(v, (int, float, np.floating, np.integer)):
        v = float(v)
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.2f}"
    return str(v)


def format_money(v):
    if v < 0:
        return f"-${abs(v):,.0f}"
    return f"${v:,.0f}"


def safe_preset(values, meta):
    """Clamp a preset dict to the valid options/ranges from feature_meta.json."""
    safe = {}
    for col in meta["features"]:
        v = values.get(col)
        if col in meta["categorical"]:
            options = meta["categories"][col]
            safe[col] = v if v in options else options[0]
        else:
            r = meta["ranges"][col]
            try:
                num = float(v) if v is not None else (r["min"] + r["max"]) / 2
                num = max(r["min"], min(num, r["max"]))
                safe[col] = num
            except (TypeError, ValueError):
                safe[col] = (r["min"] + r["max"]) / 2
    return safe


PRESET_LABEL_TO_KEY = {
    "Low risk": "low_risk",
    "Typical": "typical",
    "High risk": "high_risk",
}


def sidebar_inputs(meta):
    st.sidebar.header("Borrower features")

    preset_label = st.sidebar.pills(
        "Try a preset",
        options=list(PRESET_LABEL_TO_KEY.keys()),
        selection_mode="single",
        default="Typical",
        key="preset_select",
    )

    # Apply preset whenever the selection changes (including the initial render).
    prev = st.session_state.get("_prev_preset")
    if preset_label != prev:
        st.session_state._prev_preset = preset_label
        if preset_label is not None:
            safe = safe_preset(PRESETS[PRESET_LABEL_TO_KEY[preset_label]], meta)
            for col, v in safe.items():
                st.session_state[f"input_{col}"] = v

    st.sidebar.divider()

    inputs = {}
    for col in meta["features"]:
        label = col.replace("_", " ").title()
        wkey = f"input_{col}"
        if col in meta["categorical"]:
            options = meta["categories"][col]
            inputs[col] = st.sidebar.selectbox(label, options, key=wkey)
        else:
            r = meta["ranges"][col]
            step = (r["max"] - r["min"]) / 100 if r["max"] > r["min"] else 1.0
            inputs[col] = st.sidebar.slider(
                label,
                min_value=float(r["min"]),
                max_value=float(r["max"]),
                step=float(step),
                key=wkey,
            )
    return inputs


def render_probability_indicator(prob, baseline_prob):
    if prob < 0.10:
        color = "#2ca02c"
    elif prob < 0.25:
        color = "#e89914"
    else:
        color = "#d62728"

    pct = min(max(prob * 100, 0), 100)
    baseline_pct = min(max(baseline_prob * 100, 0), 100)

    html = f"""
    <div style="text-align: center; padding: 8px 0 4px;">
      <div style="font-size: 13px; color: #888; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600;">
        Probability of default
      </div>
      <div style="font-size: 64px; font-weight: 600; color: {color}; line-height: 1.1; margin-top: 4px;">
        {prob:.1%}
      </div>
    </div>
    <div style="position: relative; height: 14px; margin: 18px 4px 8px;
                background: linear-gradient(to right,
                  #2ca02c 0%, #2ca02c 10%,
                  #e89914 10%, #e89914 25%,
                  #d62728 25%, #d62728 100%);
                border-radius: 7px; opacity: 0.55;">
      <div style="position: absolute; left: {pct:.2f}%; top: -6px;
                  width: 4px; height: 26px; background: #0a2540;
                  border-radius: 2px; transform: translateX(-2px);
                  box-shadow: 0 1px 4px rgba(0,0,0,0.3);"></div>
      <div style="position: absolute; left: {baseline_pct:.2f}%; top: 16px;
                  font-size: 11px; color: #888; transform: translateX(-50%); white-space: nowrap;">
        ▲ baseline {baseline_prob:.0%}
      </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #888; margin: 4px 4px 0;">
      <span>0%</span><span>10%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def calc_economics(loan_amnt, installment, term, prob_default):
    expected_interest_if_paid = max(installment * term - loan_amnt, 0.0)
    expected_loss_if_default = 0.5 * loan_amnt
    expected_value = (
        (1 - prob_default) * expected_interest_if_paid
        - prob_default * expected_loss_if_default
    )
    return {
        "principal": loan_amnt,
        "expected_interest_if_paid": expected_interest_if_paid,
        "expected_loss_if_default": expected_loss_if_default,
        "expected_value": expected_value,
        "is_profitable": expected_value > 0,
    }


def find_top_improvements(model, X, meta, base_prob, top_n=3):
    rows = []
    feature_for_row = []
    suggested_for_row = []
    current_row = X.iloc[0]

    for col in meta["features"]:
        if col in meta["categorical"]:
            options = meta["categories"][col]
            current = current_row[col]
            for opt in options:
                if opt == current:
                    continue
                row = current_row.copy()
                row[col] = opt
                rows.append(row)
                feature_for_row.append(col)
                suggested_for_row.append(opt)
        else:
            r = meta["ranges"][col]
            grid = np.linspace(r["min"], r["max"], 20)
            for v in grid:
                row = current_row.copy()
                row[col] = v
                rows.append(row)
                feature_for_row.append(col)
                suggested_for_row.append(v)

    if not rows:
        return []

    X_alt = pd.DataFrame(rows).reset_index(drop=True)
    for cat_col in meta["categorical"]:
        X_alt[cat_col] = pd.Categorical(
            X_alt[cat_col], categories=meta["categories"][cat_col]
        )

    probs = model.predict_proba(X_alt)[:, 1]

    best_per_feature = {}
    for i, col in enumerate(feature_for_row):
        if col not in best_per_feature or probs[i] < best_per_feature[col]["new_prob"]:
            best_per_feature[col] = {
                "feature": col,
                "suggested": suggested_for_row[i],
                "current": current_row[col],
                "new_prob": float(probs[i]),
            }

    improvements = []
    for col, info in best_per_feature.items():
        delta = base_prob - info["new_prob"]
        if delta > 0.005:
            info["delta"] = delta
            improvements.append(info)

    improvements.sort(key=lambda x: x["delta"], reverse=True)
    return improvements[:top_n]


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

    st.title("Loan Default Predictor")
    st.write(
        "LightGBM model trained on the Lending Club dataset. Try a preset "
        "borrower in the sidebar, then move the sliders. The default probability, "
        "the per-feature contribution chart, the loan economics, and the top "
        "single-feature changes that would lower risk all update live."
    )

    inputs = sidebar_inputs(meta)

    X = pd.DataFrame([inputs])
    for col in meta["categorical"]:
        X[col] = pd.Categorical(X[col], categories=meta["categories"][col])

    prob = float(model.predict_proba(X)[0, 1])

    contribs = model.predict(X, pred_contrib=True)
    feature_contribs = contribs[0, :-1]
    bias = float(contribs[0, -1])
    baseline_prob = 1.0 / (1.0 + np.exp(-bias))

    render_probability_indicator(prob, baseline_prob)

    if prob < 0.10:
        st.success("**APPROVE** — low risk, well below approval threshold")
    elif prob < 0.25:
        st.info("**MANUAL REVIEW** — moderate risk, judgment call")
    else:
        st.warning("**DECLINE** — elevated risk above the typical approval threshold")

    st.write("")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Loan economics")
        loan_amnt = float(X.iloc[0]["loan_amnt"])
        installment = float(X.iloc[0]["installment"])
        term = float(X.iloc[0]["term"])
        econ = calc_economics(loan_amnt, installment, term, prob)

        c1, c2 = st.columns(2)
        c1.metric("Principal", format_money(econ["principal"]))
        c2.metric(
            "Interest if repaid",
            format_money(econ["expected_interest_if_paid"]),
        )

        c3, c4 = st.columns(2)
        c3.metric(
            "Loss if default (50% LGD)",
            format_money(econ["expected_loss_if_default"]),
        )
        c4.metric(
            "Net expected value",
            format_money(econ["expected_value"]),
            delta="profitable" if econ["is_profitable"] else "loss",
            delta_color="normal" if econ["is_profitable"] else "inverse",
        )

        st.caption(
            f"EV = (1 − {prob:.1%}) × interest − {prob:.1%} × loss. "
            "50% loss-given-default is a rough industry approximation; "
            "tune it to your portfolio's actual recovery rate."
        )

    with right:
        st.subheader("Why this prediction")

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

        st.caption(
            "Each bar is the log-odds contribution from that specific feature "
            "value. Red pushes toward default, green toward repayment. "
            "Sum + baseline = model output."
        )

    st.subheader("What single change would help most?")
    st.caption(
        "Brute-force search: for each feature, what value (holding the rest "
        "constant) gives the lowest default probability?"
    )

    improvements = find_top_improvements(model, X, meta, prob, top_n=3)

    if improvements:
        cols = st.columns(3)
        for i, imp in enumerate(improvements):
            with cols[i]:
                st.markdown(f"**{imp['feature'].replace('_', ' ').title()}**")
                st.markdown(
                    f"`{format_value(imp['current'])}` → "
                    f"`{format_value(imp['suggested'])}`"
                )
                st.metric(
                    "New default probability",
                    f"{imp['new_prob']:.1%}",
                    delta=f"-{imp['delta']:.1%}",
                    delta_color="inverse",
                )
    else:
        st.info(
            "No single-feature change drops risk meaningfully for this borrower. "
            "Either they're already low-risk or the model views them as "
            "structurally elevated across all individual features."
        )

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
