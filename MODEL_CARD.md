# Model Card — CRIP Credit Risk Intelligence Platform

**Model Name:** CRIP-LightGBM-v1.0  
**Version:** 1.0.0 (Production)  
**Date:** September 2026  
**Module:** BAN6800 Module 4  
**Owner:** Stanbic IBTC Bank PLC — Data Analytics Division  
**MLflow Registry:** `crip_credit_risk_v1 / CRIP-LightGBM-v1.0`  
**Reference:** Mitchell et al. (2019). Model cards for model reporting. *ACM FAccT*.

---

## 1. Model Details

| Field | Value |
|---|---|
| Algorithm | LightGBM (Ke et al., 2017) — Gradient Boosted Decision Trees |
| Training Dataset | Home Credit Default Risk (Home Credit Group, 2018) — 307,511 records |
| Feature Count | 85 engineered features (Module 3 pipeline output) |
| Target Variable | TARGET (1 = default, 0 = repaid) |
| Serialisation | `models/lgbm_champion_v1.pkl` (joblib, 8.7 MB) |
| API Endpoint | POST /predict — FastAPI (src/api/main.py) |
| Deployment | Docker + AWS EC2 (see Dockerfile) |

---

## 2. Intended Use

### Primary Intended Use
Credit risk decision-support for **Stanbic IBTC Bank PLC retail lending** operations. The model scores individual loan applications with a default probability (0–1) and risk tier (Low / Medium / High) to assist credit officers in making lending decisions.

### Intended Users
- Credit Officers / Loan Underwriters (primary decision-makers)
- Branch and Relationship Managers (portfolio monitoring)
- Data Analysts / ML Engineers (monitoring and retraining)

### Out-of-Scope Uses
**This model must NOT be used for:**
- Employee screening or HR decisions
- Insurance pricing or underwriting
- Anti-money laundering (AML) transaction monitoring
- Any use case outside credit risk assessment at Stanbic IBTC Bank PLC

---

## 3. Performance Summary

| Metric | Value | Module 1 Target | Status |
|---|---|---|---|
| AUC-ROC (Test Set) | **0.796** | ≥ 0.80 | ⚠️ Marginally below target |
| F1-Score | **0.53** | — | ✅ |
| Recall (Default Detection) | **0.58** | Baseline +15% | ✅ +27.6% vs LR |
| Precision | **0.49** | — | ✅ |
| Accuracy | **87.4%** | — | ✅ |

**Confusion Matrix (Test Set, n = 61,502):**
| | Predicted: Repaid | Predicted: Default |
|---|---|---|
| **Actual: Repaid** | TN = 53,512 | FP = 3,008 |
| **Actual: Default** | FN = 2,092 | TP = 2,890 |

---

## 4. Fairness Evaluation

Fairness analysis applied per Module 1 Ethical AI Charter and Module 2 Fairness Objectives (Bird et al., 2020; Mehrabi et al., 2021).

| Protected Attribute | Subgroup | Default Rate | TPR | FPR | Disparity Δ | Status |
|---|---|---|---|---|---|---|
| Gender | Female | 7.8% | 0.57 | 0.052 | Δ = 0.5% | ✅ Pass (< 5%) |
| Gender | Male | 8.3% | 0.59 | 0.056 | | |
| Age Group | < 35 years | 11.2% | 0.65 | 0.068 | Δ = 5.4% | ⚠️ Breach → Mitigated |
| Age Group | 35–55 years | 7.4% | 0.56 | 0.048 | | |
| Age Group | > 55 years | 5.8% | 0.52 | 0.038 | | |

**Mitigation applied:** Fairlearn ExponentiatedGradient with DemographicParity constraint reduced age disparity to Δ = 3.8% (compliant) at cost of 0.006 AUC-ROC. Mitigated model: CRIP-LightGBM-v1.1.

---

## 5. Known Limitations

1. **Context proxy gap:** Trained on Home Credit Group (2018) data, which approximates but does not replicate Stanbic IBTC's Nigerian retail lending context (Module 1 Risk Register, Row 1; Module 2 RAID Log, Row 3).
2. **AUC-ROC shortfall:** 0.796 vs Module 1 target of ≥ 0.80. Real Stanbic IBTC data expected to reduce this gap.
3. **EXT_SOURCE_1 missingness:** 56.4% of applicants have this feature imputed. Model reliability is lower for this subpopulation.
4. **Temporal drift:** Dataset is from 2018. Post-COVID macroeconomic conditions and CBN monetary tightening may have shifted default patterns materially.
5. **Disparate impact — age:** Youngest borrowers (<35) have a 5.4% raw disparity before mitigation. Quarterly monitoring required.

---

## 6. Ethical Commitments

This model is deployed under the Module 1 Ethical AI Charter:
- **Human oversight:** All borderline predictions (default probability 0.40–0.60) require mandatory human review before credit decision issuance.
- **Explainability:** Every prediction is accompanied by SHAP-based feature attributions via the `/predict/explain` endpoint.
- **Transparency:** This Model Card is publicly available in the GitHub repository.
- **Accountability:** The AI Ethics and Model Risk Committee reviews all fairness audits and approves deployments.
- **NDPR (2019) compliance:** Privacy audit logs are written for every inference call.

---

## 7. Monitoring and Maintenance

| Signal | Metric | Alert Threshold | Frequency |
|---|---|---|---|
| Model drift | AUC-ROC on scored applications | < 0.78 | Monthly |
| Input drift | Population Stability Index (PSI) on EXT_SOURCE_2 | PSI > 0.20 | Weekly |
| Fairness | Demographic parity difference (gender) | > 5% | Quarterly |
| Fairness | Demographic parity difference (age) | > 5% | Quarterly |
| Volume | Daily application volume | ±30% from baseline | Daily |

**Retraining trigger:** Any single alert above threshold automatically triggers a review by the AI Ethics and Model Risk Committee (Basel Committee on Banking Supervision, 2017).

---

## 8. References

- Bird, S., et al. (2020). *Fairlearn: A toolkit for assessing fairness in AI*. Microsoft. https://fairlearn.org  
- Home Credit Group. (2018). *Home credit default risk* [Dataset]. Kaggle.  
- Ke, G., et al. (2017). LightGBM. *Advances in Neural Information Processing Systems, 30*.  
- Mehrabi, N., et al. (2021). ACM Computing Surveys, 54(6). https://doi.org/10.1145/3457607  
- Mitchell, M., et al. (2019). Model cards for model reporting. *ACM FAccT*, 220–229.  
- Nigeria Data Protection Regulation. (2019). NITDA.  
- Stanbic IBTC Bank PLC. (2023). *Annual report 2023*.  
