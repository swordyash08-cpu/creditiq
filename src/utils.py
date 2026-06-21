import streamlit as st
import numpy as np

# ── EMI Calculator ─────────────────────────────────────────────────────────────

def calculate_emi(principal, annual_rate, tenure_months):
    """EMI = P·r·(1+r)^n / ((1+r)^n − 1)"""
    if principal <= 0 or annual_rate <= 0 or tenure_months <= 0:
        return 0.0
    r = (annual_rate / 12.0) / 100.0
    n = tenure_months
    emi = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    return round(emi, 2)


def format_inr(amount):
    """Formats a number as Indian currency with ₹ symbol and commas."""
    if amount >= 10_000_000:
        return f"₹{amount/10_000_000:.2f} Cr"
    elif amount >= 100_000:
        return f"₹{amount/100_000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


# ── Risk Grade UI helpers ──────────────────────────────────────────────────────

GRADE_COLORS = {
    'A+': '#10b981',  # Emerald
    'A':  '#34d399',  # Green
    'B+': '#f59e0b',  # Amber
    'B':  '#fb923c',  # Orange
    'C':  '#ef4444',  # Red
    'D':  '#dc2626',  # Deep Red
}

GRADE_BG = {
    'A+': 'rgba(16,185,129,0.15)',
    'A':  'rgba(52,211,153,0.12)',
    'B+': 'rgba(245,158,11,0.15)',
    'B':  'rgba(251,146,60,0.15)',
    'C':  'rgba(239,68,68,0.15)',
    'D':  'rgba(220,38,38,0.18)',
}

def get_grade_color(grade):
    return GRADE_COLORS.get(grade, '#94a3b8')

def get_grade_bg(grade):
    return GRADE_BG.get(grade, 'rgba(100,116,139,0.15)')


def dti_classification(dti):
    if dti <= 30:
        return 'Low Risk', '#10b981'
    elif dti <= 50:
        return 'Medium Risk', '#f59e0b'
    else:
        return 'High Risk', '#ef4444'


# ── Custom CSS ─────────────────────────────────────────────────────────────────

def inject_custom_css():

    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(145deg, #060b18 0%, #0a1120 40%, #0f172a 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06091a 0%, #0a1020 50%, #0d1627 100%) !important;
        border-right: 1px solid rgba(99,102,241,0.18) !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-size: 0.88rem;
        padding: 5px 0;
        transition: color 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover { color: #e2e8f0 !important; }

    /* ── Glassmorphism cards ── */
    .card {
        background: rgba(11, 18, 36, 0.75);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 4px 32px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
        margin-bottom: 20px;
        transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
    }
    .card:hover {
        border-color: rgba(99,102,241,0.30);
        box-shadow: 0 8px 48px rgba(99,102,241,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    /* ── KPI Metric cards ── */
    .kpi-card {
        background: rgba(11,18,36,0.80);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        backdrop-filter: blur(16px);
        margin-bottom: 16px;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,0.5), transparent);
        opacity: 0;
        transition: opacity 0.3s;
    }
    .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 36px rgba(99,102,241,0.18); border-color: rgba(99,102,241,0.25); }
    .kpi-card:hover::before { opacity: 1; }
    .kpi-label { font-size: 10px; letter-spacing: 1.8px; text-transform: uppercase; color: #475569; font-weight: 600; margin-bottom: 8px; }
    .kpi-value { font-size: 28px; font-weight: 800; line-height: 1; margin-bottom: 4px; }
    .kpi-sub   { font-size: 11px; color: #64748b; margin-top: 5px; }

    /* ── Decision banners ── */
    .banner {
        border-radius: 14px; padding: 20px 26px; margin-bottom: 22px;
        font-size: 18px; font-weight: 700; border-left: 5px solid;
        animation: fadeIn 0.4s ease;
    }
    .banner-approved   { background: rgba(16,185,129,0.10);  color: #10b981; border-color: #10b981; }
    .banner-conditional{ background: rgba(245,158,11,0.10);  color: #f59e0b; border-color: #f59e0b; }
    .banner-manual     { background: rgba(59,130,246,0.10);  color: #60a5fa; border-color: #3b82f6; }
    .banner-rejected   { background: rgba(239,68,68,0.10);   color: #ef4444; border-color: #ef4444; }

    /* ── Hard stop box ── */
    .hard-stop {
        background: rgba(239,68,68,0.07); border: 1px dashed rgba(239,68,68,0.60);
        border-radius: 10px; padding: 14px 18px; color: #fca5a5;
        font-size: 0.88rem; margin-bottom: 12px;
    }

    /* ── Red flag / positive items ── */
    .flag-item {
        background: rgba(245,158,11,0.07); border-left: 3px solid #f59e0b;
        padding: 8px 14px; border-radius: 6px; color: #fcd34d;
        font-size: 0.86rem; margin-bottom: 8px;
    }
    .pos-item {
        background: rgba(16,185,129,0.07); border-left: 3px solid #10b981;
        padding: 8px 14px; border-radius: 6px; color: #6ee7b7;
        font-size: 0.86rem; margin-bottom: 8px;
    }

    /* ── Analyst note ── */
    .analyst-note {
        background: rgba(15,23,42,0.65); border: 1px solid rgba(99,102,241,0.22);
        border-radius: 12px; padding: 18px 22px; color: #cbd5e1;
        font-size: 0.91rem; line-height: 1.75; font-style: italic;
    }

    /* ── Stress test card ── */
    .stress-card {
        background: rgba(11,18,36,0.60); border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; padding: 16px 18px; text-align: center; margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .stress-card:hover { border-color: rgba(99,102,241,0.25); }

    /* ── Improvement suggestion ── */
    .suggestion-item {
        background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.18);
        border-radius: 10px; padding: 12px 16px; color: #a7f3d0;
        font-size: 0.88rem; margin-bottom: 10px;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
        border-radius: 8px !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        padding: 12px 32px !important; font-weight: 700 !important;
        font-size: 0.95rem !important; letter-spacing: 0.5px !important;
        transition: all 0.22s ease !important;
        box-shadow: 0 4px 18px rgba(99,102,241,0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(99,102,241,0.50) !important;
    }

    /* ── Headers ── */
    h1 {
        background: linear-gradient(90deg, #e2e8f0 30%, #94a3b8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800 !important; letter-spacing: -0.5px;
    }
    h2, h3, h4 { color: #e2e8f0 !important; font-weight: 700 !important; }

    /* ── Input widgets ── */
    .stSlider > div > div > div > div { background: #6366f1 !important; }
    .stSelectbox > div > div { background: rgba(11,18,36,0.80) !important; border-color: rgba(99,102,241,0.28) !important; }
    .stNumberInput > div > div > input { background: rgba(11,18,36,0.80) !important; color: #e2e8f0 !important; border-color: rgba(99,102,241,0.25) !important; }

    /* ── Dividers ── */
    hr { border-color: rgba(99,102,241,0.12) !important; }

    /* ── Section header ── */
    .section-header {
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6366f1;
        font-weight: 700;
        margin-bottom: 12px;
        margin-top: 4px;
    }

    /* ── Chat Interface ── */
    .chat-container {
        max-height: 520px;
        overflow-y: auto;
        padding: 12px 4px;
        scrollbar-width: thin;
        scrollbar-color: rgba(99,102,241,0.25) transparent;
    }
    .chat-container::-webkit-scrollbar { width: 5px; }
    .chat-container::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 4px; }
    .chat-user {
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.30);
        border-radius: 14px 14px 4px 14px;
        padding: 12px 16px;
        margin: 8px 0 8px 40px;
        color: #e2e8f0;
        font-size: 0.90rem;
        line-height: 1.6;
    }
    .chat-assistant {
        background: rgba(11,18,36,0.85);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px 14px 14px 4px;
        padding: 14px 18px;
        margin: 8px 40px 8px 0;
        color: #cbd5e1;
        font-size: 0.90rem;
        line-height: 1.7;
    }
    .chat-quick-btn {
        display: inline-block;
        background: rgba(99,102,241,0.10);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 20px;
        padding: 6px 14px;
        color: #a5b4fc;
        font-size: 0.78rem;
        cursor: pointer;
        transition: all 0.2s;
        margin: 3px;
    }
    .chat-quick-btn:hover {
        background: rgba(99,102,241,0.25);
        border-color: rgba(99,102,241,0.50);
    }

    /* ── Upload Wizard ── */
    .upload-zone {
        background: rgba(99,102,241,0.06);
        border: 2px dashed rgba(99,102,241,0.30);
        border-radius: 16px;
        padding: 40px 24px;
        text-align: center;
        transition: all 0.3s;
    }
    .upload-zone:hover {
        border-color: rgba(99,102,241,0.60);
        background: rgba(99,102,241,0.10);
    }
    .upload-step {
        background: rgba(11,18,36,0.70);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .upload-step-active {
        border-color: rgba(99,102,241,0.40);
        background: rgba(99,102,241,0.08);
    }
    .upload-step-done {
        border-color: rgba(16,185,129,0.30);
        background: rgba(16,185,129,0.06);
    }
    .quality-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* ── Search Panel ── */
    .search-result-card {
        background: rgba(11,18,36,0.70);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
        transition: border-color 0.2s;
        cursor: pointer;
    }
    .search-result-card:hover {
        border-color: rgba(99,102,241,0.30);
    }
    .filter-chip {
        display: inline-block;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.75rem;
        color: #a5b4fc;
        margin: 2px;
    }

    /* ── Policy Editor ── */
    .policy-card {
        background: rgba(11,18,36,0.75);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        transition: border-color 0.2s;
    }
    .policy-card:hover {
        border-color: rgba(99,102,241,0.25);
    }
    .policy-changed {
        border-color: rgba(245,158,11,0.50) !important;
        background: rgba(245,158,11,0.06) !important;
    }

    /* ── Scenario Cards ── */
    .scenario-card {
        background: rgba(11,18,36,0.75);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: all 0.25s;
        position: relative;
        overflow: hidden;
    }
    .scenario-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99,102,241,0.30);
        box-shadow: 0 8px 32px rgba(99,102,241,0.15);
    }
    .scenario-severity-moderate {
        border-left: 4px solid #f59e0b;
    }
    .scenario-severity-severe {
        border-left: 4px solid #ef4444;
    }
    .scenario-severity-critical {
        border-left: 4px solid #dc2626;
    }
    .impact-delta-positive {
        color: #ef4444;
        font-weight: 800;
    }
    .impact-delta-negative {
        color: #10b981;
        font-weight: 800;
    }

    /* ── Animations ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.5; }
    }
    .typing-indicator {
        display: inline-flex;
        gap: 4px;
        padding: 8px 14px;
    }
    .typing-indicator span {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #6366f1;
        animation: pulse 1.2s infinite;
    }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.20); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.40); }

    /* ── Dataframe styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Chat input override ── */
    .stChatInput > div { border-color: rgba(99,102,241,0.30) !important; }
    .stChatInput > div:focus-within { border-color: rgba(99,102,241,0.60) !important; }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
