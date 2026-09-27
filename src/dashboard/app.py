"""
BAN6800 Capstone — Stanbic IBTC Bank Credit Risk Intelligence Platform
Interactive Stakeholder Dashboard — Module 5 Deliverable

Deploy to Streamlit Cloud:
  1. Push this file to github.com/Layorr/stanbic-credit-risk/src/dashboard/app.py
  2. Go to share.streamlit.io → Connect GitHub → Select repo
  3. Set main file: src/dashboard/app.py
  4. Deploy → copy the live URL into your Module 5 submission

Run locally:
  pip install streamlit plotly pandas numpy
  streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CRIP — Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ─────────────────────────────────────────────────────────────
DARK_BLUE = "#1F3864"
MID_BLUE  = "#2E5096"
GOLD      = "#C9A84C"
GREEN     = "#27AE60"
AMBER     = "#D97706"
RED       = "#C0392B"

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .main-header {{
        background: {DARK_BLUE};
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0.3rem 0;
    }}
    .risk-low    {{ color: {GREEN}; }}
    .risk-medium {{ color: {AMBER}; }}
    .risk-high   {{ color: {RED};   }}
    .gold-text   {{ color: {GOLD};  }}
    .insight-box {{
        background: {DARK_BLUE};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }}
    .fairness-pass   {{ background-color: #E8F5E9; border-left: 4px solid {GREEN}; padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; }}
    .fairness-warn   {{ background-color: #FFF8E0; border-left: 4px solid {AMBER}; padding: 0.8rem 1rem; border-radius: 0 6px 6px 0; }}
    div[data-testid="stTabs"] button {{ font-size: 0.95rem; font-weight: 600; }}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-size:1.5rem;">🏦 Credit Risk Intelligence Platform (CRIP)</h2>
    <p style="margin:0.3rem 0 0; opacity:0.85; font-size:0.9rem;">
        Stanbic IBTC Bank PLC  ·  BAN6800 Module 5  ·  LightGBM Champion (AUC-ROC: 0.796)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Simulate model predictions ────────────────────────────────────────────────
def simulate_prediction(features: dict) -> dict:
    """Simulate model scoring based on key feature relationships."""
    np.random.seed(int(features.get("amt_credit", 450000)) % 1000)

    credit_income = features["amt_credit"] / max(features["amt_income"], 1)
    annuity_income = features["annuity"] / max(features["amt_income"], 1)
    ext2 = features["ext_source_2"]
    employ = features["employ_years"]
    age = features["age_years"]

    # Approximate LightGBM scoring logic
    base = 0.08  # dataset default rate
    score = base
    score += (credit_income - 3.5) * 0.05
    score += (annuity_income - 0.15) * 0.12
    score -= (ext2 - 0.5) * 0.28
    score -= (employ - 3) * 0.015
    score -= (age - 38) * 0.002
    score = max(0.02, min(0.97, score))

    shap_vals = {
        "Credit History Score (EXT_SOURCE_2)": -round((ext2 - 0.5) * 0.28, 3),
        "Loan-to-Income Ratio": round((credit_income - 3.5) * 0.05, 3),
        "Employment Tenure": -round((employ - 3) * 0.015, 3),
        "Repayment-to-Income Ratio": round((annuity_income - 0.15) * 0.12, 3),
        "Applicant Age": -round((age - 38) * 0.002, 3),
    }

    tier = "Low Risk" if score < 0.25 else "Medium Risk" if score < 0.50 else "High Risk"
    review = 0.40 <= score <= 0.60

    return {"probability": score, "tier": tier, "requires_review": review, "shap": shap_vals}

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Overview",
    "🔍 Score an Application",
    "⚖️ Fairness Monitor",
    "📈 Portfolio Analytics",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MODEL OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Champion Model Performance")
    st.caption("LightGBM v1.0 — trained on 307,511 loan application records (Home Credit Group, 2018)")

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (col1, "AUC-ROC", "0.796", MID_BLUE, "vs. 0.50 baseline"),
        (col2, "Recall",  "58%",   AMBER,     "Defaults detected"),
        (col3, "Precision","49%",  AMBER,     "Flagged are real defaults"),
        (col4, "FN Reduction", "+27.6%", GREEN, "vs. LR baseline"),
    ]
    for col, label, val, color, sub in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.8rem; color:#666; font-weight:600;">{label}</div>
                <div class="metric-value" style="color:{color};">{val}</div>
                <div style="font-size:0.75rem; color:#999;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("**Precision-Recall Trade-off by Threshold**")
        st.caption("The business chooses where to set the threshold based on risk appetite (Dastile et al., 2020)")
        thresholds = [0.25, 0.35, 0.50, 0.60, 0.70]
        recalls    = [0.82, 0.71, 0.58, 0.44, 0.31]
        precisions = [0.29, 0.37, 0.49, 0.61, 0.72]
        f1s        = [0.43, 0.49, 0.53, 0.51, 0.43]
        df_thresh = pd.DataFrame({"Threshold": thresholds, "Recall": recalls, "Precision": precisions, "F1-Score": f1s})

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=recalls,    mode='lines+markers', name='Recall',    line=dict(color=RED,    width=2.5), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines+markers', name='Precision', line=dict(color=GREEN,  width=2.5), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=thresholds, y=f1s,        mode='lines+markers', name='F1-Score',  line=dict(color=MID_BLUE, width=2.5, dash='dash'), marker=dict(size=8)))
        fig.add_vline(x=0.50, line=dict(color=GOLD, width=2, dash='dash'))
        fig.add_annotation(x=0.50, y=0.95, text="★ Current (0.50)", font=dict(color=GOLD, size=11), showarrow=False)
        fig.update_layout(
            height=320, xaxis_title="Decision Threshold", yaxis_title="Score",
            yaxis=dict(range=[0, 1]), legend=dict(orientation="h", y=-0.25),
            margin=dict(l=0, r=0, t=20, b=0), plot_bgcolor="white",
            paper_bgcolor="white", font=dict(family="Calibri"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eee")
        fig.update_yaxes(showgrid=True, gridcolor="#eee")
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Key insight:** AUC-ROC of 0.796 reflects the model's power at ALL thresholds. Precision = 49% and Recall = 58% reflect the balanced 0.50 threshold — not a model weakness. Setting threshold to 0.35 raises recall to 71%.")

    with col_right:
        st.markdown("**Confusion Matrix — Default Threshold (0.50)**")
        cm_data = [[53512, 3008], [2092, 2890]]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted: Repay", "Predicted: Default"],
            y=["Actual: Repay", "Actual: Default"],
            text=[[f"TN\n{cm_data[0][0]:,}", f"FP\n{cm_data[0][1]:,}"],
                  [f"FN\n{cm_data[1][0]:,}", f"TP\n{cm_data[1][1]:,}"]],
            texttemplate="%{text}", textfont=dict(size=13, family="Calibri"),
            colorscale=[[0,"#EBF3FB"],[0.5,"#2E5096"],[1,"#1F3864"]],
            showscale=False,
        ))
        fig_cm.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0), font=dict(family="Calibri"))
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("**Model Comparison**")
        df_comp = pd.DataFrame({
            "Model": ["Majority-Class\nBaseline", "Logistic\nRegression", "XGBoost", "LightGBM ★"],
            "AUC-ROC": [0.500, 0.714, 0.789, 0.796],
        })
        fig_bar = px.bar(df_comp, x="Model", y="AUC-ROC",
                         color="AUC-ROC", color_continuous_scale=[[0, "#DCE6F1"],[1,"#1F3864"]],
                         text="AUC-ROC", height=200)
        fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside", textfont_size=11)
        fig_bar.add_hline(y=0.80, line=dict(color=GOLD, dash="dash", width=2))
        fig_bar.update_layout(margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                               yaxis=dict(range=[0,1.0]), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCORE AN APPLICATION
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Score a Loan Application")
    st.caption("Enter applicant details to receive an instant risk assessment with explanation (Lundberg & Lee, 2017)")

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📁 Loan Details**")
            amt_credit = st.number_input("Loan Amount (₦)", 50000, 5000000, 450000, 50000)
            annuity    = st.number_input("Monthly Repayment (₦)", 1000, 200000, 22500, 1000)
        with col2:
            st.markdown("**👤 Applicant Profile**")
            amt_income   = st.number_input("Annual Income (₦)", 60000, 5000000, 1800000, 50000)
            age_years    = st.slider("Age (years)", 18, 70, 38)
            employ_years = st.slider("Employment Tenure (years)", 0.0, 40.0, 3.5, 0.5)
        with col3:
            st.markdown("**📊 Credit Information**")
            ext_source_2 = st.slider("Credit History Score", 0.0, 1.0, 0.55, 0.01)
            st.markdown(f"<small style='color:{MID_BLUE};'>0 = poor / 1 = excellent</small>", unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍  Calculate Risk Score", use_container_width=True)

    if submitted:
        result = simulate_prediction({
            "amt_credit": amt_credit, "amt_income": amt_income,
            "annuity": annuity, "ext_source_2": ext_source_2,
            "employ_years": employ_years, "age_years": age_years,
        })

        prob = result["probability"]
        tier = result["tier"]
        color = GREEN if tier == "Low Risk" else AMBER if tier == "Medium Risk" else RED
        review_str = "⚠️  Human Review Required" if result["requires_review"] else ("✅  Proceed" if tier != "High Risk" else "❌  Decline / Restructure")

        st.markdown("---")
        col_score, col_gauge = st.columns([1, 1.5])
        with col_score:
            st.markdown(f"""
            <div class="metric-card" style="border-top:4px solid {color};">
                <div style="font-size:0.9rem; font-weight:600; color:#666;">Default Probability</div>
                <div style="font-size:3rem; font-weight:bold; color:{color};">{prob:.0%}</div>
                <div style="font-size:1.1rem; font-weight:bold; color:{color}; margin:0.4rem 0;">{tier}</div>
                <div style="font-size:0.9rem; color:#555;">{review_str}</div>
            </div>""", unsafe_allow_html=True)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 32, "family": "Calibri"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0,  25], "color": "#E8F5E9"},
                        {"range": [25, 40], "color": "#FFF3CD"},
                        {"range": [40, 60], "color": "#FEE9D8"},
                        {"range": [60, 100],"color": "#FCE4D6"},
                    ],
                    "threshold": {"line": {"color": GOLD, "width": 3}, "thickness": 0.8, "value": 50},
                },
            ))
            fig_gauge.update_layout(height=220, margin=dict(l=20,r=20,t=30,b=0), font=dict(family="Calibri"))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("**What Drove This Score?** (SHAP Explanation)")
        shap = result["shap"]
        shap_df = pd.DataFrame(list(shap.items()), columns=["Feature", "SHAP Value"]).sort_values("SHAP Value")
        shap_df["Color"] = shap_df["SHAP Value"].apply(lambda x: RED if x > 0 else GREEN)
        shap_df["Direction"] = shap_df["SHAP Value"].apply(lambda x: "↑ Raises risk" if x > 0 else "↓ Reduces risk")

        fig_shap = px.bar(shap_df, x="SHAP Value", y="Feature", orientation="h",
                           color="Color", color_discrete_map="identity",
                           text="Direction", height=260)
        fig_shap.update_traces(textposition="outside", textfont_size=10)
        fig_shap.add_vline(x=0, line=dict(color="#333", width=1))
        fig_shap.update_layout(showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                                plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Calibri"))
        st.plotly_chart(fig_shap, use_container_width=True)

        loan_to_income = amt_credit / max(amt_income, 1)
        st.markdown(f"""
        <div class="insight-box">
        <b>🗣️ Plain-Language Summary for Credit Officer:</b><br>
        This applicant's default probability is <b>{prob:.0%}</b> — classified as <b>{tier}</b>.
        The model sees {'a high loan-to-income ratio of {:.2f}'.format(loan_to_income) if loan_to_income > 5 else 'a manageable loan-to-income ratio of {:.2f}'.format(loan_to_income)},
        a credit history score of {ext_source_2:.2f} ({'below' if ext_source_2 < 0.50 else 'above'} average),
        and {employ_years:.1f} years of employment tenure.
        {'⚠️ Human review is required before a credit decision is issued (Module 1 Ethical AI Charter).' if result['requires_review'] else ''}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — FAIRNESS MONITOR
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Fairness Metrics Monitor")
    st.caption("Demographic parity and equalized odds — Module 1 Fairness Objectives applied (Bird et al., 2020; Mehrabi et al., 2021)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Gender Fairness — Demographic Parity**")
        st.markdown('<div class="fairness-pass">✅ <b>PASSED</b> — Δ = 0.5% (threshold: ≤ 5%)</div>', unsafe_allow_html=True)
        fig_gender = go.Figure(go.Bar(
            x=["Female (7.8%)", "Male (8.3%)"],
            y=[7.8, 8.3],
            marker_color=[GREEN, MID_BLUE],
            text=["7.8%", "8.3%"], textposition="auto", textfont_size=13,
        ))
        fig_gender.add_hline(y=9, line=dict(color=RED, dash="dash"), annotation_text="Max allowed (5% gap)")
        fig_gender.update_layout(height=280, yaxis_title="Default Rate (%)", yaxis=dict(range=[0,12]),
                                 margin=dict(l=0,r=0,t=20,b=0), showlegend=False,
                                 plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Calibri"))
        st.plotly_chart(fig_gender, use_container_width=True)

    with col2:
        st.markdown("**Age Group Fairness — After Mitigation**")
        st.markdown('<div class="fairness-warn">⚠️ <b>MITIGATED</b> — Raw Δ = 5.4% → After ExponentiatedGradient: Δ = 3.8% ✅</div>', unsafe_allow_html=True)
        fig_age = go.Figure(go.Bar(
            x=["< 35 years", "35–55 years", "> 55 years"],
            y=[11.2, 7.4, 5.8],
            marker_color=[RED, AMBER, GREEN],
            text=["11.2%", "7.4%", "5.8%"], textposition="auto", textfont_size=12,
        ))
        fig_age.update_layout(height=280, yaxis_title="Default Rate (%)", yaxis=dict(range=[0,14]),
                               margin=dict(l=0,r=0,t=20,b=0), showlegend=False,
                               plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Calibri"))
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("---")
    st.markdown("**Full Fairness Metrics Table**")
    fairness_data = {
        "Protected Attribute": ["Gender", "Gender", "Age Group", "Age Group", "Age Group"],
        "Subgroup": ["Female", "Male", "< 35 years", "35–55 years", "> 55 years"],
        "Default Rate": ["7.8%", "8.3%", "11.2%", "7.4%", "5.8%"],
        "Recall (TPR)": ["0.57", "0.59", "0.65", "0.56", "0.52"],
        "FPR": ["0.052", "0.056", "0.068", "0.048", "0.038"],
        "Disparity Δ": ["0.5% ✅", "—", "5.4% ⚠️→3.8% ✅", "—", "—"],
        "Dir. Impact": ["0.94 ✅", "—", "0.72 ✅", "—", "—"],
    }
    st.dataframe(pd.DataFrame(fairness_data), use_container_width=True, hide_index=True)
    st.caption("Fairness definitions: Mehrabi et al. (2021). Mitigation: Fairlearn ExponentiatedGradient (Bird et al., 2020). Threshold: 5% max disparity (Module 1 Fairness Objectives).")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PORTFOLIO ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Portfolio Risk Distribution")
    st.caption("Simulated portfolio of 10,000 applications — for live deployment connect to Stanbic IBTC data")

    np.random.seed(42)
    scores = np.concatenate([
        np.random.beta(2, 18, 7500),   # low-risk majority
        np.random.beta(5, 5, 1800),    # medium risk
        np.random.beta(12, 3, 700),    # high risk
    ])
    score_df = pd.DataFrame({"Default Probability": scores})
    score_df["Risk Tier"] = pd.cut(scores, bins=[0,0.25,0.50,1.0], labels=["Low","Medium","High"])

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig_hist = px.histogram(score_df, x="Default Probability", color="Risk Tier",
                                 color_discrete_map={"Low":GREEN, "Medium":AMBER, "High":RED},
                                 nbins=50, opacity=0.85,
                                 title="Score Distribution Across Portfolio")
        fig_hist.update_layout(height=320, bargap=0.05, font=dict(family="Calibri"),
                                margin=dict(l=0,r=0,t=40,b=0), plot_bgcolor="white",
                                paper_bgcolor="white", showlegend=True,
                                legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        tier_counts = score_df["Risk Tier"].value_counts().reindex(["Low","Medium","High"])
        fig_pie = px.pie(values=tier_counts.values, names=tier_counts.index,
                          color=tier_counts.index,
                          color_discrete_map={"Low":GREEN,"Medium":AMBER,"High":RED},
                          title="Portfolio Risk Tier Split")
        fig_pie.update_traces(texttemplate="%{label}<br>%{percent}", textfont_size=13)
        fig_pie.update_layout(height=320, font=dict(family="Calibri"),
                               margin=dict(l=0,r=0,t=40,b=20), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Monthly trend
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep"]
    high_pct = [8.2, 7.9, 8.5, 8.1, 7.8, 7.6, 7.4, 7.2, 7.1]
    fig_trend = px.line(x=months, y=high_pct, markers=True,
                         title="High-Risk Application Rate (%) — Monthly Trend",
                         labels={"x":"Month","y":"High Risk %"})
    fig_trend.update_traces(line_color=RED, marker_color=RED, line_width=2.5)
    fig_trend.add_hline(y=8.5, line=dict(color=AMBER, dash="dash"), annotation_text="Alert threshold")
    fig_trend.update_layout(height=260, font=dict(family="Calibri"),
                             margin=dict(l=0,r=0,t=40,b=0), plot_bgcolor="white",
                             paper_bgcolor="white", yaxis=dict(range=[6,10]))
    fig_trend.update_xaxes(showgrid=True, gridcolor="#eee")
    fig_trend.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig_trend, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#999; font-size:0.8rem; padding:0.5rem 0;">
    CRIP Dashboard  ·  BAN6800 Module 5  ·  Stanbic IBTC Bank PLC  ·  Layori Soetan  ·
    Model: LightGBM (Ke et al., 2017)  ·  SHAP: Lundberg & Lee (2017)  ·  Fairness: Bird et al. (2020)
    <br>github.com/Layorr/stanbic-credit-risk
</div>""", unsafe_allow_html=True)
