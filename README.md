# 🛡️ CreditIQ — AI-Powered Credit Underwriting & Risk Intelligence Platform

> **"Bloomberg Terminal meets Stripe"** — A production-grade AI underwriting decision-support platform for credit analysts, risk managers, and credit committees.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditiq.streamlit.app)

---

## 🏦 Platform Overview

CreditIQ is an enterprise-grade AI-assisted credit underwriting platform that mirrors software used inside HDFC Bank, ICICI Bank, Bajaj Finance, and RBI-regulated NBFCs. It combines a **Rule-Based Scorecard** with a **Random Forest ML Engine** in a dual-layer decisioning architecture.

### 7 Pages — Full Underwriting Suite

| Page | Description |
|------|-------------|
| 📊 **Portfolio Dashboard** | 10 KPIs — AUM, Default Rate, NPA, Provision Coverage, CIBIL avg |
| 🛡️ **Credit Assessment** | Full 5 Cs underwriting + XAI + Stress Testing + PDF Memo |
| 🎛️ **What-If Simulator** | Real-time score dial + Fastest Approval Path |
| ⚖️ **Model Governance** | AUC-ROC, 5-Fold CV, Disagreement Audit, Weight Justification |
| 🎓 **Knowledge Base** | FAQ, Glossary, Dataset Governance, Methodology |
| 📄 **Credit Memo Generator** | Formal PDF Credit Committee Memorandum |
| 🕵️ **Decision Audit Trail** | Immutable session log with CSV export |

---

## 🔬 Technical Architecture

```
CreditIQ
├── app.py                  # Main 7-page Streamlit application
├── src/
│   ├── model.py            # Dual-layer engine: Rule-Based + Random Forest
│   ├── utils.py            # CSS design system + EMI/Grade helpers
│   ├── memo.py             # FPDF2 Credit Committee Memo generator
│   └── generator.py        # Synthetic 8,000-borrower portfolio dataset
├── requirements.txt
└── .streamlit/config.toml  # Dark banking theme
```

### Dual-Layer Decision Engine
- **Layer 1 — Rule-Based Scorecard**: 5 Cs of Credit framework (Character 35%, Capacity 30%, Capital 15%, Collateral 15%, Conditions 5%) with hard-stop policy rules
- **Layer 2 — Random Forest ML**: Trained on 8,000 synthetic borrowers, AUC-ROC > 0.86, 5-Fold Cross-Validated
- **Consensus Engine**: Detects model agreement/disagreement, escalates conflicts to manual review

### Risk Metrics
| Metric | Value |
|--------|-------|
| AUC-ROC | > 0.86 |
| 5-Fold CV AUC | 0.8549 ± 0.006 |
| Training Records | 8,000 |
| Risk Grades | A+ → D (6 tiers) |
| Policy Rules | 6 hard-stop checks |

---

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/creditiq.git
cd creditiq
pip install -r requirements.txt
streamlit run app.py
```

---

## 📐 Credit Frameworks Implemented

- **5 Cs of Credit** — Character, Capacity, Capital, Collateral, Conditions
- **Basel III** — PD × LGD × EAD Expected Loss calculation
- **RBI NBFC Guidelines** — Hard-stop thresholds, DTI limits, LTV ratios
- **Model Governance** — Champion-challenger framework, drift detection, audit trail

---

## ⚠️ Disclaimer

CreditIQ is an AI-assisted underwriting decision-support **prototype** developed as an MBA Finance Capstone Live Project. All data is **synthetic**. This does not constitute a binding lending decision, regulatory compliance opinion, or financial advice.

---

*MBA Finance Capstone · Rule-Based Scorecard + Random Forest · 5 Cs Framework · Dual-Layer AI Engine*
