"""Streamlit demo for the loan default predictor."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ARTIFACTS = Path("artifacts")

st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="📊",
    layout="wide",
)

# ---------- design system ----------
ACCENT = "#0a2540"
GREEN = "#2ca02c"
AMBER = "#e89914"
RED = "#d62728"
MUTED = "#525252"
SUBTLE = "#737373"
PANEL_BORDER = "#e8e6df"

CUSTOM_CSS = """
<style>
/* Tighter top padding on the main content */
.block-container { padding-top: 2.5rem !important; padding-bottom: 4rem !important; }

/* Section spacing */
h2, h3 { margin-top: 1.6rem !important; }

/* Subhead labels */
.section-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #737373;
  margin-bottom: 4px;
}

/* Card containers */
.card {
  background: #ffffff;
  border: 1px solid #e8e6df;
  border-radius: 12px;
  padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(10,10,10,0.03);
}

/* Verdict pill */
.verdict {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  border-radius: 999px;
  font-weight: 600;
  font-size: 14px;
  letter-spacing: 0.01em;
}
.verdict.approve { background: #e8f5e9; color: #1b5e20; border: 1px solid #c5e8c8; }
.verdict.review  { background: #fff8e1; color: #8b6914; border: 1px solid #f5e1a4; }
.verdict.decline { background: #ffebee; color: #b71c1c; border: 1px solid #f5c2c2; }

/* Counterfactual cards */
.cf-card {
  background: #ffffff;
  border: 1px solid #e8e6df;
  border-radius: 12px;
  padding: 16px 18px;
  height: 100%;
}
.cf-feature { font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #737373; margin-bottom: 6px; }
.cf-change { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; color: #525252; margin-bottom: 12px; }
.cf-value { font-size: 28px; font-weight: 600; color: #0a2540; line-height: 1; }
.cf-delta { font-size: 13px; color: #2ca02c; margin-top: 4px; font-weight: 600; }

/* Percentile bars */
.pct-row { display: flex; align-items: center; gap: 14px; margin-bottom: 8px; padding: 6px 0; }
.pct-label { width: 130px; font-size: 13px; color: #525252; font-weight: 500; }
.pct-track { flex: 1; height: 8px; background: #f0eee8; border-radius: 4px; position: relative; overflow: hidden; }
.pct-fill  { position: absolute; top: 0; bottom: 0; left: 0; border-radius: 4px; }
.pct-marker { position: absolute; top: -3px; width: 3px; height: 14px; background: #0a2540; border-radius: 1.5px; transform: translateX(-1.5px); }
.pct-text { width: 130px; font-size: 12px; color: #525252; text-align: right; font-variant-numeric: tabular-nums; }

/* Tighten the metric text a bit */
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 600 !important; color: #0a2540 !important; }
[data-testid="stMetricLabel"] { font-size: 12px !important; font-weight: 500 !important; color: #737373 !important; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: #efece4 !important;
  border-right: 1px solid #e0ddd4;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
</style>
"""


# Hand-tuned borrower profiles. All values are clamped to valid options/ranges
# at runtime via safe_preset(), so if the trained model's category lists or
# ranges differ slightly from these guesses the app still works.
PRESETS = {
    "low_risk": {
        "loan_amnt": 7000.0, "term": "36", "int_rate": 7.5, "installment": 220.0,
        "grade": "A", "emp_length": 10.0, "home_ownership": "MORTGAGE",
        "annual_inc": 95000.0, "verification_status": "Source Verified",
        "purpose": "credit_card", "addr_state": "CA", "dti": 9.0,
        "delinq_2yrs": 0.0, "fico_range_low": 770.0, "inq_last_6mths": 0.0,
        "open_acc": 14.0, "pub_rec": 0.0, "revol_bal": 5000.0,
        "revol_util": 15.0, "total_acc": 30.0,
    },
    "typical": {
        "loan_amnt": 12000.0, "term": "36", "int_rate": 13.0, "installment": 400.0,
        "grade": "C", "emp_length": 5.0, "home_ownership": "MORTGAGE",
        "annual_inc": 65000.0, "verification_status": "Source Verified",
        "purpose": "debt_consolidation", "addr_state": "CA", "dti": 18.0,
        "delinq_2yrs": 0.0, "fico_range_low": 690.0, "inq_last_6mths": 0.0,
        "open_acc": 11.0, "pub_rec": 0.0, "revol_bal": 12000.0,
        "revol_util": 50.0, "total_acc": 24.0,
    },
    "high_risk": {
        "loan_amnt": 28000.0, "term": "60", "int_rate": 22.0, "installment": 770.0,
        "grade": "E", "emp_length": 1.0, "home_ownership": "RENT",
        "annual_inc": 38000.0, "verification_status": "Not Verified",
        "purpose": "debt_consolidation", "addr_state": "FL", "dti": 32.0,
        "delinq_2yrs": 2.0, "fico_range_low": 645.0, "inq_last_6mths": 3.0,
        "open_acc": 8.0, "pub_rec": 1.0, "revol_bal": 18000.0,
        "revol_util": 88.0, "total_acc": 14.0,
    },
}


PRESET_LABEL_TO_KEY = {
    "Low risk": "low_risk",
    "Typical": "typical",
    "High risk": "high_risk",
}


@st.cache_resource
def load_model():
    return joblib.load(ARTIFACTS / "model.pkl")


@st.cache_data
def load_json(name: str):
    path = ARTIFACTS / name
    if not path.exists():
        return {}
    with open(path) as f:
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


def sidebar_inputs(meta):
    st.sidebar.header("Borrower features")

    preset_label = st.sidebar.pills(
        "Try a preset",
        options=list(PRESET_LABEL_TO_KEY.keys()),
        selection_mode="single",
        default="Typical",
        key="preset_select",
    )

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


def probability_gauge(prob, baseline_prob):
    """Plotly indicator: half-circle gauge with risk zones and a baseline marker."""
    if prob < 0.10:
        bar_color = GREEN
    elif prob < 0.25:
        bar_color = AMBER
    else:
        bar_color = RED

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={
                "suffix": "%",
                "valueformat": ".1f",
                "font": {"size": 56, "color": ACCENT, "family": "Arial"},
            },
            gauge={
                "axis": {
                    "range": [0, 60],
                    "tickwidth": 1,
                    "tickcolor": SUBTLE,
                    "tickfont": {"size": 11, "color": SUBTLE},
                    "ticksuffix": "%",
                },
                "bar": {"color": bar_color, "thickness": 0.32},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 10], "color": "#e8f5e9"},
                    {"range": [10, 25], "color": "#fff8e1"},
                    {"range": [25, 60], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": ACCENT, "width": 3},
                    "thickness": 0.85,
                    "value": baseline_prob * 100,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(
        height=260,
        margin={"t": 20, "b": 0, "l": 30, "r": 30},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial", "color": ACCENT},
    )
    return fig


def contribution_chart(model, X, feature_contribs, top_n=10):
    """Plotly horizontal bar of per-feature contributions, with hover and clean labels."""
    contrib_df = pd.DataFrame(
        {
            "feature": model.feature_name_,
            "contribution": feature_contribs,
            "value": [format_value(X.iloc[0][f]) for f in model.feature_name_],
        }
    )
    contrib_df["abs"] = contrib_df["contribution"].abs()
    top = contrib_df.nlargest(top_n, "abs").sort_values("contribution")

    labels = [
        f"<b>{row['feature'].replace('_', ' ')}</b><br>"
        f"<span style='color:{SUBTLE};font-size:11px;'>{row['value']}</span>"
        for _, row in top.iterrows()
    ]
    colors = [RED if c > 0 else GREEN for c in top["contribution"]]

    fig = go.Figure(
        go.Bar(
            x=top["contribution"],
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Value: %{customdata[1]}<br>"
                "Contribution: %{x:+.3f} log-odds<extra></extra>"
            ),
            customdata=list(
                zip(
                    top["feature"].str.replace("_", " "),
                    top["value"],
                )
            ),
        )
    )
    fig.add_vline(x=0, line_color=SUBTLE, line_width=1)
    fig.update_layout(
        height=420,
        margin={"t": 8, "b": 50, "l": 4, "r": 16},
        xaxis={
            "title": "Contribution to log-odds of default",
            "title_font": {"size": 12, "color": MUTED},
            "tickfont": {"size": 11, "color": MUTED},
            "gridcolor": "#f0eee8",
            "zerolinecolor": SUBTLE,
        },
        yaxis={"tickfont": {"size": 12, "color": ACCENT}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font={"family": "Arial"},
    )
    return fig


def percentile_rank(value, stats):
    """Approximate percentile of `value` from p5/p25/p50/p75/p95 anchors."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    breakpoints = [
        (stats["p5"], 5), (stats["p25"], 25), (stats["p50"], 50),
        (stats["p75"], 75), (stats["p95"], 95),
    ]
    if value <= stats["p5"]:
        return 5
    if value >= stats["p95"]:
        return 95
    for i in range(len(breakpoints) - 1):
        v_low, p_low = breakpoints[i]
        v_high, p_high = breakpoints[i + 1]
        if v_low <= value <= v_high:
            if v_high == v_low:
                return p_low
            ratio = (value - v_low) / (v_high - v_low)
            return p_low + ratio * (p_high - p_low)
    return 50


def render_percentile_panel(X, pop_stats):
    """For each tracked feature, draw a horizontal bar showing borrower's rank vs population."""
    show_features = [
        ("fico_range_low", "FICO score"),
        ("annual_inc", "Annual income"),
        ("dti", "Debt-to-income"),
        ("revol_util", "Revolving utilization"),
        ("loan_amnt", "Loan amount"),
        ("int_rate", "Interest rate"),
    ]

    rows_html = []
    for col, label in show_features:
        if col not in pop_stats:
            continue
        stats = pop_stats[col]
        value = X.iloc[0].get(col)
        try:
            pct = percentile_rank(float(value), stats)
        except (TypeError, ValueError):
            pct = None
        if pct is None:
            continue

        lower_better = stats.get("lower_is_better", False)
        # Color the fill by where the borrower sits
        if lower_better:
            # low percentile = good (green)
            if pct < 33:
                fill_color = GREEN
            elif pct < 66:
                fill_color = AMBER
            else:
                fill_color = RED
        else:
            # high percentile = good (green)
            if pct > 66:
                fill_color = GREEN
            elif pct > 33:
                fill_color = AMBER
            else:
                fill_color = RED

        # Format value for display
        display_value = format_value(value)
        if col == "annual_inc" or col == "loan_amnt" or col == "revol_bal" or col == "installment":
            display_value = format_money(float(value))
        elif col in ("int_rate", "revol_util", "dti"):
            display_value = f"{float(value):.1f}%"

        # Format percentile description
        rank_text = f"P{pct:.0f} of all borrowers"

        row = f"""
        <div class="pct-row">
          <div class="pct-label">{label}</div>
          <div class="pct-track">
            <div class="pct-fill" style="background: {fill_color}; opacity: 0.35; width: {pct:.1f}%;"></div>
            <div class="pct-marker" style="left: {pct:.1f}%;"></div>
          </div>
          <div class="pct-text">{display_value} · {rank_text}</div>
        </div>
        """
        rows_html.append(row)

    if rows_html:
        st.markdown("".join(rows_html), unsafe_allow_html=True)


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
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if not (ARTIFACTS / "model.pkl").exists():
        st.error(
            "No trained model found. Run `python train.py` first to generate "
            "`artifacts/model.pkl`."
        )
        return

    model = load_model()
    meta = load_json("feature_meta.json")
    metrics = load_json("metrics.json")
    pop_stats = load_json("population_stats.json")

    st.title("Loan Default Predictor")
    st.write(
        "LightGBM model trained on the Lending Club dataset. Try a preset "
        "borrower in the sidebar, then move the sliders. The default "
        "probability, the per-feature contribution chart, the population "
        "percentile rank panel, the loan economics, and the top "
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

    # ===== Decision header =====
    head_left, head_right = st.columns([3, 2])

    with head_left:
        st.markdown('<div class="section-label">Risk gauge</div>', unsafe_allow_html=True)
        st.plotly_chart(probability_gauge(prob, baseline_prob), use_container_width=True, theme=None)

    with head_right:
        st.markdown('<div class="section-label">Verdict</div>', unsafe_allow_html=True)
        if prob < 0.10:
            verdict_html = '<div class="verdict approve">✓ APPROVE — low risk</div>'
            sub = "Below the typical approval threshold. Strong borrower."
        elif prob < 0.25:
            verdict_html = '<div class="verdict review">⚠ MANUAL REVIEW — moderate risk</div>'
            sub = "Mid-band. Underwriter judgment call."
        else:
            verdict_html = '<div class="verdict decline">✗ DECLINE — elevated risk</div>'
            sub = "Above the typical approval threshold."

        st.markdown(verdict_html, unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{MUTED}; font-size:14px; margin-top:10px;">{sub}</div>',
            unsafe_allow_html=True,
        )
        delta = prob - baseline_prob
        direction = "above" if delta >= 0 else "below"
        st.markdown(
            f'<div style="color:{SUBTLE}; font-size:13px; margin-top:14px;">'
            f"Model baseline default rate: <b>{baseline_prob:.0%}</b>. "
            f"This borrower is <b>{abs(delta):.1%} {direction}</b> baseline."
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ===== Loan economics + Per-feature contributions =====
    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="section-label">Loan economics</div>', unsafe_allow_html=True)
        st.subheader("Profit math")
        loan_amnt = float(X.iloc[0]["loan_amnt"])
        installment = float(X.iloc[0]["installment"])
        term = float(X.iloc[0]["term"])
        econ = calc_economics(loan_amnt, installment, term, prob)

        c1, c2 = st.columns(2)
        c1.metric("Principal", format_money(econ["principal"]))
        c2.metric("Interest if repaid", format_money(econ["expected_interest_if_paid"]))

        c3, c4 = st.columns(2)
        c3.metric("Loss if default (50% LGD)", format_money(econ["expected_loss_if_default"]))
        c4.metric(
            "Net expected value",
            format_money(econ["expected_value"]),
            delta="profitable" if econ["is_profitable"] else "loss",
            delta_color="normal" if econ["is_profitable"] else "inverse",
        )

        st.caption(
            f"EV = (1 − {prob:.1%}) × interest − {prob:.1%} × loss. "
            "50% loss-given-default is a rough industry approximation."
        )

    with right:
        st.markdown('<div class="section-label">Why this prediction</div>', unsafe_allow_html=True)
        st.subheader("Top feature contributions")
        st.plotly_chart(
            contribution_chart(model, X, feature_contribs, top_n=10),
            use_container_width=True,
            theme=None,
        )
        st.caption(
            "Red bars push toward default, green toward repayment. Sum + "
            "baseline = model output (log-odds)."
        )

    st.divider()

    # ===== Population percentile rank panel =====
    if pop_stats:
        st.markdown('<div class="section-label">Cohort context</div>', unsafe_allow_html=True)
        st.subheader("How does this borrower rank in the population?")
        st.caption(
            "Percentile across all 1.3M settled Lending Club loans. The colored "
            "fill shows how favorable the percentile is for default risk: "
            "green = good, red = concerning. The dark bar marks this borrower's spot."
        )
        render_percentile_panel(X, pop_stats)

        st.divider()

    # ===== Counterfactuals =====
    st.markdown('<div class="section-label">What-if</div>', unsafe_allow_html=True)
    st.subheader("Single change that would help most")
    st.caption(
        "Brute-force search: for each feature, what value (holding the rest "
        "constant) gives the lowest default probability? Top three reductions."
    )

    improvements = find_top_improvements(model, X, meta, prob, top_n=3)

    if improvements:
        cols = st.columns(3)
        for i, imp in enumerate(improvements):
            with cols[i]:
                feature_name = imp["feature"].replace("_", " ").title()
                cur = format_value(imp["current"])
                sug = format_value(imp["suggested"])
                card_html = f"""
                <div class="cf-card">
                  <div class="cf-feature">{feature_name}</div>
                  <div class="cf-change"><code>{cur}</code> → <code>{sug}</code></div>
                  <div class="cf-value">{imp['new_prob']:.1%}</div>
                  <div class="cf-delta">↓ {imp['delta']:.1%} default risk</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.info(
            "No single-feature change drops risk meaningfully. Either this "
            "borrower is already low-risk, or risk is structurally elevated "
            "across all individual features."
        )

    # ===== Diagnostics =====
    with st.expander("Model details and diagnostics"):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Test AUC:** {metrics.get('auc', 'n/a')}")
            st.write(f"**Accuracy at 0.5 threshold:** {metrics.get('accuracy', 'n/a')}")
            st.write(f"**Default rate (test):** {metrics.get('default_rate_test', 'n/a')}")
            st.write(f"**Test set size:** {metrics.get('n_test', 'n/a')}")
            st.write(f"**Features used:** {metrics.get('n_features', 'n/a')}")
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
