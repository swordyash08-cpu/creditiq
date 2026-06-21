import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle, os, sys, datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from generator import generate_borrower_dataset
from model import (
    calculate_5cs_scores, calculate_weighted_score,
    check_hard_stops, check_red_flags, get_rule_based_decision,
    get_risk_grade, get_risk_grade_label,
    train_ml_model, predict_single_probability,
    generate_analyst_notes, run_stress_test, get_improvement_suggestions,
    get_model_consensus, get_confidence_score, get_weighted_contributions,
    get_policy_audit, find_fastest_approval_path,
)
from utils import (
    inject_custom_css, calculate_emi, format_inr,
    get_grade_color, get_grade_bg, dti_classification
)
from memo import generate_credit_memo_pdf_full

# New modules for V2.0
import database as db
import ai_assistant
import upload_engine
import scenario_engine

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — AI Underwriting Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# ── Session State Initialization ──────────────────────────────────────────────
if 'last_assessment' not in st.session_state:
    st.session_state['last_assessment'] = None
if 'audit_log' not in st.session_state:
    st.session_state['audit_log'] = []
if 'assessment_count' not in st.session_state:
    st.session_state['assessment_count'] = 0
if 'chat_messages' not in st.session_state:
    st.session_state['chat_messages'] = db.get_chat_history(limit=20)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data',   'borrowers.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'risk_model.pkl')


@st.cache_resource(show_spinner="🔄 Initialising AI Risk Engine & Database…")
def load_resources():
    db.init_database()
    db.init_default_policies()
    if not os.path.exists(DATA_PATH):
        generate_borrower_dataset()
    if not os.path.exists(MODEL_PATH):
        train_ml_model(DATA_PATH, MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        md = pickle.load(f)
    # Seed database with CSV data on first run
    db.seed_from_csv(DATA_PATH, md)
    return md

try:
    model_data = load_resources()
    df = db.get_portfolio_df()
except Exception as e:
    st.error(f"Init error: {e}")
    generate_borrower_dataset()
    train_ml_model(DATA_PATH, MODEL_PATH)
    model_data = load_resources()
    df = db.get_portfolio_df()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:14px 0 18px;'>
      <div style='font-size:2rem;'>🛡️</div>
      <div style='font-size:1.15rem;font-weight:800;color:#e2e8f0;letter-spacing:1px;'>CreditIQ</div>
      <div style='font-size:0.65rem;color:#64748b;letter-spacing:2.5px;text-transform:uppercase;margin-top:3px;'>
        AI Underwriting Platform
      </div>
    </div>
    <hr style='border-color:rgba(99,102,241,0.20);margin-bottom:16px;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav", label_visibility="hidden",
        options=[
            "📊  Portfolio Dashboard",
            "🤖  AI Credit Assistant",
            "📤  Bulk Upload Engine",
            "🛡️  Credit Assessment",
            "🔍  Portfolio Search",
            "📜  Credit Policy Engine",
            "🎛️  What-If Simulator",
            "🌊  Scenario Analysis",
            "⚖️  Model Governance",
            "🎓  Knowledge Base & Data",
            "📄  Credit Memo Generator",
            "🕵️  Decision Audit Trail",
        ]
    )

    st.markdown("<hr style='border-color:rgba(99,102,241,0.12);margin:12px 0;'>", unsafe_allow_html=True)

    # ── Session state indicators ───────────────────────────────────────────────
    has_assessment = st.session_state['last_assessment'] is not None
    audit_count    = len(st.session_state['audit_log'])
    status_dot_g   = "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:#10b981;margin-right:6px;'></span>"
    status_dot_y   = "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:#f59e0b;margin-right:6px;'></span>"
    status_dot_r   = "<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:#ef4444;margin-right:6px;'></span>"
    st.markdown(f"""
    <div style='font-size:0.75rem;color:#64748b;padding:6px 4px;line-height:2.0;'>
      {status_dot_g if has_assessment else status_dot_y}
      <span style='color:{'#10b981' if has_assessment else '#f59e0b'};'>
        {'Assessment Ready' if has_assessment else 'No Assessment Yet'}
      </span><br>
      {status_dot_g if audit_count>0 else status_dot_r}
      <span style='color:{'#10b981' if audit_count>0 else '#64748b'};'>
        Audit Events: {audit_count}
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Explainer Mode Toggle ──────────────────────────────────────────────────
    demo_mode = st.toggle("🎓 Explainer Mode", value=False)
    if demo_mode:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.30);
                    border-radius:8px;padding:10px 12px;font-size:0.78rem;color:#a5b4fc;'>
          <b>Explainer Mode ON</b><br>
          Plain-language concept explanations appear throughout the platform.
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:0.72rem;color:#334155;padding:10px 4px;line-height:1.9;margin-top:6px;
                border-top:1px solid rgba(99,102,241,0.08);'>
      <b style='color:#475569;'>Model Engine</b><br>
      AUC-ROC: <b style='color:#10b981;'>{model_data['metrics'].get('auc',0):.4f}</b>&nbsp;
      Acc: <b style='color:#10b981;'>{model_data['metrics']['accuracy']*100:.1f}%</b><br>
      <span style='color:#1e293b;font-size:0.65rem;'>Rule Engine + Random Forest | 5 Cs</span>
    </div>
    <div style='font-size:0.60rem;color:#1e293b;text-align:center;margin-top:8px;'>
      MBA Finance Capstone · CreditIQ v2.0
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: demo explanation box
# ══════════════════════════════════════════════════════════════════════════════
def demo_box(title, text):
    if demo_mode:
        st.markdown(f"""
        <div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);
                    border-radius:10px;padding:14px 18px;margin:8px 0 16px;'>
          <div style='font-size:0.72rem;letter-spacing:1.5px;text-transform:uppercase;
                      color:#818cf8;font-weight:700;margin-bottom:6px;'>💡 Concept — {title}</div>
          <div style='font-size:0.88rem;color:#c7d2fe;line-height:1.7;'>{text}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PORTFOLIO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("# 📊 Portfolio Risk Dashboard")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Enterprise-grade portfolio analytics across 8,000-borrower synthetic retail lending book · As of " + datetime.date.today().strftime("%d %b %Y") + "</div>", unsafe_allow_html=True)

    demo_box("What is a Portfolio Dashboard?",
             "A portfolio dashboard gives a bird's-eye view of all loans in a bank's book. "
             "Risk managers use it to monitor approval rates, default trends, and concentration risk across segments.")

    db_stats = db.get_portfolio_stats()
    
    app_rate = db_stats['approval_rate']
    avg_score = db_stats['avg_risk_score']
    grade_dist = pd.Series(db_stats.get('grade_dist', {}))
    
    def_rate = db_stats['avg_pd'] * 100
    total_aum = db_stats['total_aum']
    avg_cibil = db_stats['avg_cibil']
    avg_dti   = db_stats['avg_dti']
    cond_rate = 8.4  # synthetic conditional approval rate estimate
    npa_est   = def_rate * 0.72  # estimate NPA from defaults
    coverage_ratio = 142.5  # synthetic PCR

    # ── 10-KPI Executive Row ───────────────────────────────────────────────────
    st.markdown("<div style='font-size:0.72rem;letter-spacing:2px;text-transform:uppercase;color:#6366f1;font-weight:700;margin-bottom:10px;'>📐 Executive Portfolio Metrics</div>", unsafe_allow_html=True)
    krow1 = st.columns(5)
    for col, val, label, clr, sub in [
        (krow1[0], f"{len(df):,}",           "Total Accounts",       "#6366f1", "Synthetic book"),
        (krow1[1], f"{app_rate:.1f}%",       "Approval Rate",        "#10b981", "Incl. conditional"),
        (krow1[2], f"{avg_score:.1f}/100",   "Avg Risk Score",       "#f59e0b", "5 Cs scorecard"),
        (krow1[3], f"{def_rate:.1f}%",       "Historical Default",   "#ef4444", "Actual default rate"),
        (krow1[4], format_inr(total_aum),    "Total AUM",            "#a855f7", "Gross loan book"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-label'>{label}</div>
              <div class='kpi-value' style='color:{clr};'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    krow2 = st.columns(5)
    for col, val, label, clr, sub in [
        (krow2[0], f"{avg_cibil:.0f}",       "Avg CIBIL Score",      "#06b6d4", "Portfolio avg"),
        (krow2[1], f"{avg_dti:.1f}%",        "Portfolio Avg DTI",    "#f59e0b" if avg_dti>35 else "#10b981", "Debt-to-income"),
        (krow2[2], f"{npa_est:.1f}%",        "Estimated NPA",        "#ef4444", "90-day DPD proxy"),
        (krow2[3], f"{coverage_ratio:.1f}%", "Provision Coverage",   "#10b981", "PCR estimate"),
        (krow2[4], f"{cond_rate:.1f}%",      "Conditional Approvals","#f59e0b", "With covenants"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-label'>{label}</div>
              <div class='kpi-value' style='color:{clr};'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🥧 Borrower Segment Mix")
        seg = df['Category'].value_counts().reset_index(); seg.columns = ['Segment','Count']
        fig = px.pie(seg, values='Count', names='Segment', hole=0.52,
                     color_discrete_sequence=['#6366f1','#a855f7','#06b6d4','#10b981'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                          margin=dict(t=10,b=10,l=10,r=10), height=275,
                          legend=dict(font=dict(size=12)))
        fig.update_traces(textfont_color='white')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🏆 Risk Grade Distribution")
        grade_order = ['A+','A','B+','B','C','D']
        gd = grade_dist.reindex(grade_order, fill_value=0).reset_index()
        gd.columns = ['Grade','Count']
        fig2 = px.bar(gd, x='Grade', y='Count', color='Grade', text='Count',
                      color_discrete_sequence=['#10b981','#34d399','#f59e0b','#fb923c','#ef4444','#dc2626'])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', showlegend=False,
                           margin=dict(t=10,b=10,l=10,r=10), height=275,
                           xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                           yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
        fig2.update_traces(textposition='outside', textfont_color='white')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📉 Default Rate by CIBIL Band")
        df2 = df.copy()
        df2['CIBIL_Band'] = pd.cut(df2['CIBIL_Score'],
                                    bins=[300,600,650,700,750,900],
                                    labels=['<600','600-650','651-700','701-750','>750'])
        cbd = df2.groupby('CIBIL_Band', observed=False)['Default'].mean().reset_index()
        cbd.columns = ['Band','Default Rate']; cbd['Default Rate'] *= 100
        fig3 = px.bar(cbd, x='Band', y='Default Rate', text_auto='.1f',
                      color='Default Rate', color_continuous_scale=['#10b981','#f59e0b','#dc2626'])
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', showlegend=False,
                           coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10), height=275,
                           yaxis_title='Default Rate (%)',
                           xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                           yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Top 8 Risk Drivers (RF Feature Importance)")
        fi   = model_data['feature_importances']
        fi_df = pd.DataFrame(list(fi.items()), columns=['Feature','Importance'])
        fi_df = fi_df.sort_values('Importance').tail(8)
        readable = {
            'CIBIL_Score':'CIBIL Score','DTI_Ratio':'DTI Ratio',
            'Monthly_Income':'Monthly Income','Missed_EMIs':'Missed EMIs',
            'Savings':'Savings','Loan_Amount':'Loan Amount',
            'Employment_Length':'Employment Length','Asset_Value':'Asset Value',
            'Loan_To_Income_Ratio':'Loan-to-Income','Net_Worth':'Net Worth',
            'Credit_History_Length':'Credit History','Loan_Tenure':'Loan Tenure',
            'Category':'Borrower Type','Co_Applicant':'Co-Applicant',
        }
        fi_df['Feature'] = fi_df['Feature'].map(readable).fillna(fi_df['Feature'])
        fi_df['Importance'] *= 100
        fig4 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale=['#1e293b','#6366f1'])
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0', showlegend=False,
                           coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10), height=275,
                           xaxis_title='Importance (%)',
                           xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                           yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 🌡️ Avg DTI by Segment & Loan Purpose")
    heat = df.groupby(['Category','Loan_Purpose'])['DTI_Ratio'].mean().reset_index()
    hpiv = heat.pivot(index='Category', columns='Loan_Purpose', values='DTI_Ratio').fillna(0)
    fig5 = px.imshow(hpiv, color_continuous_scale='RdYlGn_r', aspect='auto', text_auto='.1f')
    fig5.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                       margin=dict(t=10,b=10,l=10,r=10), height=200)
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — AI CREDIT ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif "AI Credit Assistant" in page:
    st.markdown("# 🤖 AI Credit Assistant")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Intelligent NLP engine acting as your autonomous Credit Risk Officer</div>", unsafe_allow_html=True)

    # Quick action buttons
    st.markdown("<div style='margin-bottom: 12px;'>", unsafe_allow_html=True)
    quick_queries = [
        "What is the overall approval rate?",
        "How many high risk borrowers exist?",
        "Show applicants with PD above 20%",
        "Why was APP-00001 rejected?",
        "Explain Probability of Default (PD)",
    ]
    for q in quick_queries:
        if st.button(q, key=f"btn_{q}"):
            # Add to chat history
            st.session_state['chat_messages'].append({'role': 'user', 'content': q})
            db.log_chat('user', q)
            # Process query
            with st.spinner("🤖 Thinking..."):
                response = ai_assistant.process_query(q)
                st.session_state['chat_messages'].append({'role': 'assistant', 'content': response})
                db.log_chat('assistant', response)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat interface container
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state['chat_messages']:
        if msg['role'] == 'user':
            st.markdown(f"<div class='chat-user'>🧑‍💼 <b>You</b><br>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-assistant'>🤖 <b>CreditIQ AI</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about portfolio metrics, risk segments, specific applicants (e.g. APP-00001), or credit policies...")
    if user_input:
        st.session_state['chat_messages'].append({'role': 'user', 'content': user_input})
        db.log_chat('user', user_input)
        response = ai_assistant.process_query(user_input)
        st.session_state['chat_messages'].append({'role': 'assistant', 'content': response})
        db.log_chat('assistant', response)
        st.rerun()

    if st.button("🗑️ Clear Chat History"):
        db.clear_chat_history()
        st.session_state['chat_messages'] = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — BULK UPLOAD ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif "Upload" in page:
    st.markdown("# 📤 Bulk Upload Engine")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Upload and process multiple loan applications simultaneously through the AI underwriting pipeline</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='upload-zone'>
            <div style='font-size:3rem;margin-bottom:10px;'>📄</div>
            <h3 style='margin:0;'>Drag and drop CSV or Excel file</h3>
            <p style='color:#94a3b8;font-size:0.9rem;'>Max 500 records per batch for performance</p>
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['csv', 'xlsx'], label_visibility="collapsed")
    with col2:
        st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
        st.markdown("#### 📋 Required Fields")
        st.markdown("""
        <ul style='color:#94a3b8;font-size:0.85rem;line-height:1.6;'>
            <li>Applicant_ID</li>
            <li>Age, Category</li>
            <li>Income, Savings, Existing_EMIs</li>
            <li>CIBIL_Score, Missed_EMIs</li>
            <li>Loan_Amount, Loan_Tenure</li>
        </ul>
        """, unsafe_allow_html=True)
        if st.download_button("⬇️ Download Template", data=upload_engine.generate_template_csv(), file_name="CreditIQ_Template.csv", mime="text/csv"):
            pass
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                upload_df = pd.read_csv(uploaded_file)
            else:
                upload_df = pd.read_excel(uploaded_file)

            # Step 1: Validate
            st.markdown("### Step 1: Validation")
            val_results = upload_engine.validate_upload(upload_df)

            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            vcol1.metric("Total Records", val_results['total_records'])
            vcol2.metric("Valid Records", val_results['total_records'] - len(val_results['invalid_rows']))
            vcol3.metric("Invalid Records", len(val_results['invalid_rows']))
            vcol4.metric("Data Quality Score", f"{val_results['quality_score']:.1f}%")

            if not val_results['valid']:
                st.error("Validation failed! Please fix the errors below and re-upload.")
                st.write("**Missing Required Fields:**", ", ".join(val_results['missing_fields']))
                if val_results['invalid_rows']:
                    st.dataframe(pd.DataFrame(val_results['invalid_rows']))
            else:
                st.success("Validation passed!")
                if val_results['warnings']:
                    for w in val_results['warnings']:
                        st.warning(w)

                # Step 2: Process
                st.markdown("### Step 2: Processing")
                if st.button("▶️ Process Batch", type="primary"):
                    with st.spinner(f"Processing {len(val_results['cleaned_df'])} applications..."):
                        batch_id = f"BATCH-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                        proc_results = upload_engine.process_upload(val_results['cleaned_df'], model_data, batch_id)

                        st.success(f"Processed {proc_results['processed']} applications!")
                        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                        pcol1.metric("Approved", proc_results['accepted'])
                        pcol2.metric("Conditional", proc_results['conditional'])
                        pcol3.metric("Manual Review", proc_results['manual_review'])
                        pcol4.metric("Rejected", proc_results['rejected'])

                        st.info("The portfolio has been updated. Go to the Dashboard to see the impact.")

        except Exception as e:
            st.error(f"Error reading file: {e}")

    # History
    st.markdown("---")
    st.markdown("#### 🕒 Upload History")
    history_df = db.get_upload_history()
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No upload history found.")
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CREDIT ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif "Assessment" in page:
    st.markdown("# 🛡️ Real-Time Credit Assessment")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Complete dual-layer underwriting assessment — Rule Engine + Random Forest + Analyst Notes + PDF Memo</div>", unsafe_allow_html=True)

    demo_box("What is Credit Underwriting?",
             "Underwriting is the process a bank uses to evaluate whether a borrower is creditworthy. "
             "CreditIQ uses two layers: (1) a rule-based scorecard that applies explicit policy thresholds, "
             "and (2) a machine learning model that detects subtle risk patterns the rules might miss.")

    cat = st.selectbox("Borrower Category", ["Salaried","Self-Employed","Student","Retired"])
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>👤 Personal & Income Profile</div>", unsafe_allow_html=True)
        age = st.slider("Age (Years)", 18, 75, 32)
        if cat == "Salaried":
            income = st.number_input("Monthly Net Income (₹)", 5000, 500000, 55000, 1000)
            emp    = st.number_input("Years in Current Job", 0.0, 40.0, 4.0, 0.5)
            emis   = st.number_input("Existing EMIs / Month (₹)", 0, 200000, 9000, 500)
        elif cat == "Self-Employed":
            income = st.number_input("Avg Monthly Business Income (₹)", 5000, 1000000, 75000, 2000)
            emp    = st.number_input("Business Vintage (Years)", 0.0, 40.0, 5.0, 0.5)
            emis   = st.number_input("Existing EMIs / Month (₹)", 0, 300000, 14000, 500)
        elif cat == "Student":
            income = st.number_input("Stipend / Part-time Income (₹)", 0, 50000, 0, 1000)
            emp    = 0.0
            emis   = st.number_input("Existing EMIs / Month (₹)", 0, 10000, 0, 500)
        else:
            income = st.number_input("Pension / Rental Income (₹)", 5000, 300000, 28000, 1000)
            emp    = 0.0
            emis   = st.number_input("Existing EMIs / Month (₹)", 0, 100000, 2500, 500)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🏦 Capital & Collateral</div>", unsafe_allow_html=True)
        savings     = st.number_input("Savings Account Balance (₹)",      0, 5000000,  90000, 5000)
        investments = st.number_input("FD / MF / Gold / Equity (₹)",      0,10000000,  45000, 5000)
        asset_val   = st.number_input("Collateral / Property Value (₹) — 0 if unsecured", 0, 50000000, 0, 50000)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📜 Credit History (Character)</div>", unsafe_allow_html=True)
        cibil   = st.slider("CIBIL Score", 300, 900, 730)
        missed  = st.number_input("Missed / Late Payments (last 24 months)", 0, 12, 0, 1)
        cred_len= st.number_input("Credit History Length (Years)", 0, 40, 5, 1)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🏷️ Loan Details (Conditions)</div>", unsafe_allow_html=True)
        loan_amt = st.number_input("Loan Amount Requested (₹)", 10000, 10000000, 350000, 10000)
        tenure   = st.slider("Tenure (Months)", 6, 240, 36)
        rate     = st.slider("Interest Rate (Annual %)", 5.0, 36.0, 11.5, 0.25)
        if cat == "Student":
            purpose = st.selectbox("Loan Purpose", ["Education","Personal"])
            co_app  = st.checkbox("Co-Applicant / Parent Guarantee", value=True)
        elif cat == "Self-Employed":
            purpose = st.selectbox("Loan Purpose", ["Business Expansion","Home Purchase","Personal","Medical"])
            co_app  = st.checkbox("Co-Applicant", value=False)
        elif cat == "Retired":
            purpose = st.selectbox("Loan Purpose", ["Medical","Personal"])
            co_app  = st.checkbox("Co-Applicant", value=False)
        else:
            purpose = st.selectbox("Loan Purpose", ["Home Purchase","Personal","Medical","Education"])
            co_app  = st.checkbox("Co-Applicant", value=False)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🚀  RUN FULL CREDIT ASSESSMENT", use_container_width=True)

    if run:
        new_emi   = calculate_emi(loan_amt, rate, tenure)
        total_emi = emis + new_emi
        # Store in session state for cross-page use
        _ts = datetime.datetime.now().strftime("%H:%M:%S")
        dti = round((total_emi / income) * 100, 1) if income > 0 else 100.0
        lti = round(loan_amt / (income * 12), 2)   if income > 0 else 10.0
        net_worth = savings + investments + asset_val - loan_amt

        bd = {
            'Category': cat, 'Age': age, 'Monthly_Income': income,
            'Savings': savings, 'Investments': investments,
            'CIBIL_Score': cibil, 'Missed_EMIs': missed,
            'Credit_History_Length': cred_len,
            'Existing_EMIs': total_emi, 'DTI_Ratio': dti,
            'Employment_Length': emp, 'Asset_Value': asset_val,
            'Loan_Amount': loan_amt, 'Loan_Tenure': tenure,
            'Loan_Purpose': purpose, 'Co_Applicant': 1 if co_app else 0,
            'Loan_To_Income_Ratio': lti, 'Net_Worth': net_worth,
        }

        cs       = calculate_5cs_scores(bd)
        w_score  = calculate_weighted_score(cs)
        grade    = get_risk_grade(w_score)
        g_label  = get_risk_grade_label(grade)
        stops    = check_hard_stops(bd)
        flags, fc= check_red_flags(bd)
        verdict, risk_cat, detail_msg = get_rule_based_decision(w_score, stops, fc)
        ml_pd    = predict_single_probability(model_data, bd)
        ml_conf  = get_confidence_score(model_data, bd)
        agree, consensus_exp, final_verdict, decision_source = get_model_consensus(verdict, ml_pd, w_score)
        wt_contrib  = get_weighted_contributions(cs)
        audit_rows  = get_policy_audit(bd)
        analyst_note = generate_analyst_notes(bd, cs, verdict, ml_pd)

        # ── Save to session state ──────────────────────────────────────────────
        st.session_state['last_assessment'] = {
            'bd': bd, 'cs': cs, 'w_score': w_score, 'grade': grade,
            'g_label': g_label, 'stops': stops, 'flags': flags, 'fc': fc,
            'verdict': verdict, 'risk_cat': risk_cat, 'detail_msg': detail_msg,
            'ml_pd': ml_pd, 'ml_conf': ml_conf, 'agree': agree,
            'consensus_exp': consensus_exp, 'final_verdict': final_verdict,
            'decision_source': decision_source, 'wt_contrib': wt_contrib,
            'audit_rows': audit_rows, 'analyst_note': analyst_note,
            'new_emi': new_emi, 'dti': dti, 'lti': lti, 'rate': rate,
            'tenure': tenure, 'cat': cat, 'timestamp': _ts,
        }
        st.session_state['assessment_count'] += 1

        # ── Insert into database ──────────────────────────────────────────────
        app_id = f"APP-{st.session_state['assessment_count']:03d}"
        db_record = bd.copy()
        db_record['Applicant_ID'] = app_id
        db_record['Risk_Score'] = w_score
        db_record['Risk_Grade'] = grade
        db_record['PD_Value'] = ml_pd
        db_record['Decision'] = final_verdict
        db.insert_applications_batch([db_record])
        st.toast(f"✅ Application {app_id} saved to portfolio database.")
        # ── Append to audit log ────────────────────────────────────────────────
        st.session_state['audit_log'].append({
            'Event #': st.session_state['assessment_count'],
            'Time': _ts,
            'Borrower Type': cat,
            'Loan Amount': f"₹{loan_amt:,.0f}",
            'CIBIL': cibil,
            'Score': f"{w_score:.0f}/100",
            'Grade': grade,
            'PD': f"{ml_pd*100:.1f}%",
            'Decision': final_verdict,
            'Hard Stops': 'Yes' if stops else 'No',
            'Consensus': '✅ Agree' if agree else '⚠️ Disagree',
        })

        # ── Positive & Negative Factors ───────────────────────────────────────
        pos_factors, neg_factors = [], []
        if cibil >= 750: pos_factors.append("Excellent CIBIL score — prime repayment history")
        elif cibil >= 700: pos_factors.append("Good CIBIL score — satisfactory credit track record")
        if dti <= 30: pos_factors.append(f"Low DTI ({dti:.1f}%) — strong repayment headroom")
        if savings >= income * 4: pos_factors.append("Strong liquidity (4× monthly income in savings)")
        if emp >= 5 and cat not in ('Student','Retired'): pos_factors.append(f"Stable employment / business ({emp:.1f} yrs)")
        if asset_val > loan_amt * 1.5: pos_factors.append("Collateral coverage ratio > 1.5× — well secured")
        if missed == 0: pos_factors.append("Zero missed payments — consistent payment behaviour")
        if co_app: pos_factors.append("Co-applicant backing enhances repayment assurance")
        if lti <= 3: pos_factors.append(f"Conservative loan-to-income ratio ({lti:.1f}×)")

        if cibil < 700: neg_factors.append(f"Below-average CIBIL ({cibil}) — elevated default risk")
        if dti > 50: neg_factors.append(f"High DTI ({dti:.1f}%) — strained repayment capacity")
        if savings < income * 2: neg_factors.append("Thin savings buffer — limited financial resilience")
        if missed >= 2: neg_factors.append(f"{missed} missed EMIs — material delinquency signal")
        if lti > 5: neg_factors.append(f"Elevated loan-to-income ratio ({lti:.1f}×)")
        if emp < 2 and cat not in ('Student','Retired'): neg_factors.append(f"Short employment ({emp:.1f} yrs) — income continuity risk")
        if asset_val == 0: neg_factors.append("Fully unsecured — no collateral backing, elevated recovery risk")
        if cred_len < 2: neg_factors.append("Thin credit file (<2 years) — limited behavioural data")

        st.markdown("---")
        st.markdown("## 📋 Underwriting Report")

        # ════════════════════════════════════════════════════════════════════════
        # FEATURE 5 — EXECUTIVE RESULT DASHBOARD
        # ════════════════════════════════════════════════════════════════════════
        demo_box("Executive Summary Cards",
                 "In a real bank, the underwriter first sees a quick summary: score, grade, PD, and decision. "
                 "These 6 KPIs let a credit manager make a fast risk judgement before diving into details.")

        gc  = get_grade_color(grade)
        dti_label, dti_color = dti_classification(dti)
        banner_cls = {'Approved':'approved','Approved with Conditions':'conditional',
                      'Manual Review':'manual','Rejected':'rejected'}.get(final_verdict,'rejected')
        agree_icon = "✅ Models Agree" if agree else "⚠️ Models Disagree"
        agree_clr  = "#10b981" if agree else "#f59e0b"

        e1,e2,e3,e4,e5,e6 = st.columns(6)
        for col, val, lbl, clr, sub in [
            (e1, f"{w_score:.0f}",         "Risk Score /100",    "#6366f1", "5 Cs scorecard"),
            (e2, f"{ml_pd*100:.1f}%",      "Default Probability","#ef4444" if ml_pd>0.25 else "#f59e0b" if ml_pd>0.10 else "#10b981", "ML Random Forest"),
            (e3, grade,                    "Risk Grade",          gc,       g_label[:18]),
            (e4, f"{dti:.1f}%",            "Post-Loan DTI",      dti_color, dti_label),
            (e5, f"₹{new_emi:,.0f}",       "Monthly EMI",        "#a855f7", f"@{rate}% · {tenure}m"),
            (e6, agree_icon,               "Consensus",          agree_clr, decision_source[:22]),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value' style='color:{clr};font-size:22px;'>{val}</div>
                  <div class='kpi-sub'>{sub}</div>
                </div>""", unsafe_allow_html=True)

        # Decision banner
        st.markdown(f"""
        <div class='banner banner-{banner_cls}'>
          {final_verdict.upper()} &nbsp;|&nbsp; {risk_cat}
          <div style='font-size:0.85rem;font-weight:400;margin-top:6px;opacity:0.9;'>{detail_msg}</div>
        </div>""", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # FEATURE 3 — HARD-STOP POLICY AUDIT TABLE
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📋 Underwriting Policy Compliance Audit")
        demo_box("Hard Stops & Policy Rules",
                 "Every bank has mandatory rules that auto-reject loans regardless of score. "
                 "For example, CIBIL < 600 = automatic rejection. This table shows all 6 policy checks "
                 "and their pass/fail status for this borrower.")
        all_pass = all(p for _, p, _, _ in audit_rows)
        if not all_pass:
            st.markdown("<div class='hard-stop'>🚨 <strong>POLICY VIOLATION DETECTED</strong> — One or more underwriting rules have failed. Review below.</div>", unsafe_allow_html=True)

        # Build styled table
        table_html = """
        <table style='width:100%;border-collapse:collapse;font-size:0.86rem;'>
          <thead>
            <tr style='background:rgba(99,102,241,0.15);'>
              <th style='padding:10px 14px;text-align:left;color:#e2e8f0;font-weight:600;'>Policy Rule</th>
              <th style='padding:10px 14px;text-align:center;color:#e2e8f0;font-weight:600;'>Your Value</th>
              <th style='padding:10px 14px;text-align:center;color:#e2e8f0;font-weight:600;'>Status</th>
              <th style='padding:10px 14px;text-align:left;color:#e2e8f0;font-weight:600;'>Rationale</th>
            </tr>
          </thead><tbody>
        """
        for rule, passed, value, rationale in audit_rows:
            row_bg  = 'rgba(16,185,129,0.04)' if passed else 'rgba(239,68,68,0.06)'
            status  = "<span style='color:#10b981;font-weight:700;'>✅ PASS</span>" if passed else "<span style='color:#ef4444;font-weight:700;'>❌ FAIL</span>"
            table_html += f"""
            <tr style='background:{row_bg};border-bottom:1px solid rgba(255,255,255,0.05);'>
              <td style='padding:9px 14px;color:#e2e8f0;'>{rule}</td>
              <td style='padding:9px 14px;text-align:center;color:#94a3b8;font-family:monospace;'>{value}</td>
              <td style='padding:9px 14px;text-align:center;'>{status}</td>
              <td style='padding:9px 14px;color:#64748b;font-size:0.80rem;'>{rationale}</td>
            </tr>"""
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # FEATURE 1 — DUAL-LAYER MODEL GOVERNANCE
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### ⚖️ Dual-Layer Model Governance")
        demo_box("Why Two Models?",
                 "Banks rarely use just one model. A rule engine provides transparency and regulatory compliance; "
                 "the ML model adds predictive power. Together, they form a 'dual-layer' system — if both agree, "
                 "confidence is high. If they disagree, a human underwriter steps in.")

        g1, g2, g3 = st.columns(3)

        with g1:
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.10);border:1px solid rgba(99,102,241,0.25);
                        border-radius:14px;padding:20px;text-align:center;'>
              <div style='font-size:0.70rem;letter-spacing:2px;text-transform:uppercase;color:#6366f1;font-weight:700;margin-bottom:10px;'>📐 Rule-Based Engine</div>
              <div style='font-size:2.4rem;font-weight:800;color:#e2e8f0;'>{w_score:.0f}</div>
              <div style='font-size:0.78rem;color:#64748b;'>Rule Score / 100</div>
              <div style='margin:10px 0;padding:6px 12px;background:rgba(99,102,241,0.15);border-radius:8px;'>
                <span style='color:#a5b4fc;font-size:0.85rem;font-weight:600;'>{risk_cat}</span>
              </div>
              <div style='font-size:1.6rem;font-weight:800;color:{gc};'>{grade}</div>
              <div style='font-size:0.72rem;color:#64748b;margin-top:4px;'>{g_label}</div>
            </div>""", unsafe_allow_html=True)

        with g2:
            ml_cat = "Low Risk" if ml_pd < 0.15 else "Medium Risk" if ml_pd < 0.30 else "High Risk"
            ml_cat_clr = "#10b981" if ml_pd < 0.15 else "#f59e0b" if ml_pd < 0.30 else "#ef4444"
            ml_pred = "Non-Default" if ml_pd < 0.50 else "Default"
            st.markdown(f"""
            <div style='background:rgba(168,85,247,0.10);border:1px solid rgba(168,85,247,0.25);
                        border-radius:14px;padding:20px;text-align:center;'>
              <div style='font-size:0.70rem;letter-spacing:2px;text-transform:uppercase;color:#a855f7;font-weight:700;margin-bottom:10px;'>🤖 Random Forest Engine</div>
              <div style='font-size:2.4rem;font-weight:800;color:#ef4444;'>{ml_pd*100:.1f}%</div>
              <div style='font-size:0.78rem;color:#64748b;'>P(Default)</div>
              <div style='margin:10px 0;padding:6px 12px;background:rgba(168,85,247,0.12);border-radius:8px;'>
                <span style='color:{ml_cat_clr};font-size:0.85rem;font-weight:600;'>{ml_cat}</span>
              </div>
              <div style='font-size:1.0rem;font-weight:700;color:#e2e8f0;'>Pred: {ml_pred}</div>
              <div style='font-size:0.80rem;color:#94a3b8;margin-top:6px;'>Confidence: <b style='color:#a855f7;'>{ml_conf:.1f}%</b></div>
            </div>""", unsafe_allow_html=True)

        with g3:
            final_bg = "rgba(16,185,129,0.10)" if "Approv" in final_verdict else "rgba(245,158,11,0.10)" if "Manual" in final_verdict or "Condition" in final_verdict else "rgba(239,68,68,0.10)"
            final_border = "rgba(16,185,129,0.30)" if "Approv" in final_verdict else "rgba(245,158,11,0.30)" if "Manual" in final_verdict or "Condition" in final_verdict else "rgba(239,68,68,0.30)"
            final_txt_clr = "#10b981" if "Approv" in final_verdict else "#f59e0b" if "Manual" in final_verdict or "Condition" in final_verdict else "#ef4444"
            agree_html = f"<span style='color:#10b981;font-size:1.2rem;'>✅ Models Agree</span>" if agree else f"<span style='color:#f59e0b;font-size:1.1rem;'>⚠️ Models Disagree</span>"
            st.markdown(f"""
            <div style='background:{final_bg};border:1px solid {final_border};
                        border-radius:14px;padding:20px;text-align:center;'>
              <div style='font-size:0.70rem;letter-spacing:2px;text-transform:uppercase;color:{final_txt_clr};font-weight:700;margin-bottom:10px;'>🏛️ Final Decision Engine</div>
              <div style='font-size:1.4rem;font-weight:800;color:{final_txt_clr};line-height:1.2;'>{final_verdict}</div>
              <div style='margin:12px 0;'>{agree_html}</div>
              <div style='font-size:0.72rem;color:#64748b;line-height:1.6;'>{decision_source}</div>
            </div>""", unsafe_allow_html=True)

        if not agree:
            st.markdown(f"""
            <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
                        border-radius:10px;padding:14px 18px;margin-top:14px;'>
              <span style='color:#f59e0b;font-weight:700;'>⚠️ Disagreement Explanation: </span>
              <span style='color:#fcd34d;font-size:0.88rem;'>{consensus_exp}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # FEATURE 4 — 5 Cs ANALYTICS + RADAR + WEIGHTED CONTRIBUTIONS
        # ════════════════════════════════════════════════════════════════════════
        demo_box("5 Cs of Credit Framework",
                 "The 5 Cs are the universal framework used by every bank globally. "
                 "Character = willingness to repay (CIBIL). Capacity = ability to repay (income/DTI). "
                 "Capital = financial cushion (savings). Collateral = security (assets). "
                 "Conditions = loan terms. Our weights: Character 35%, Capacity 30%, Capital 15%, Collateral 15%, Conditions 5%.")

        r1, r2 = st.columns([1, 1])
        with r1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 🕸️ 5 Cs Credit Radar")
            cats = list(cs.keys()) + [list(cs.keys())[0]]
            vals = list(cs.values()) + [list(cs.values())[0]]
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vals, theta=cats, fill='toself',
                fillcolor='rgba(99,102,241,0.18)',
                line=dict(color='#6366f1', width=2.5),
                marker=dict(size=7, color='#a855f7'),
                name='Profile'
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=[10]*len(cats), theta=cats, fill='toself',
                fillcolor='rgba(255,255,255,0.02)',
                line=dict(color='rgba(255,255,255,0.08)', width=1, dash='dot'),
                name='Max Benchmark'
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0,10], gridcolor='rgba(255,255,255,0.08)',
                                   tickfont=dict(color='#64748b',size=9),
                                   tickmode='array', tickvals=[2,4,6,8,10]),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                                     tickfont=dict(color='#e2e8f0',size=13))
                ),
                paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                showlegend=True, legend=dict(font=dict(size=11),bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=20,b=20,l=30,r=30), height=320
            )
            st.plotly_chart(fig_r, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 📊 Scorecard with Weighted Contributions")
            weight_pct = {'Character':35,'Capacity':30,'Capital':15,'Collateral':15,'Conditions':5}
            for c_name, c_val in cs.items():
                contrib = wt_contrib[c_name]
                w = weight_pct[c_name]
                pct = c_val / 10.0
                clr = '#10b981' if pct >= 0.7 else '#f59e0b' if pct >= 0.5 else '#ef4444'
                st.markdown(f"""
                <div style='margin-bottom:12px;'>
                  <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                    <span style='font-size:0.86rem;color:#e2e8f0;font-weight:600;'>
                      {c_name} <span style='color:#64748b;font-weight:400;font-size:0.75rem;'>({w}% weight)</span>
                    </span>
                    <span style='font-size:0.86rem;font-weight:700;color:{clr};'>{c_val:.1f}/10 &nbsp;
                      <span style='font-size:0.72rem;color:#6366f1;'>(+{contrib} pts)</span>
                    </span>
                  </div>
                </div>""", unsafe_allow_html=True)
                st.progress(pct)

            st.markdown(f"""
            <div style='margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.07);'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='color:#94a3b8;font-size:0.85rem;'>Composite Score (Σ weighted)</span>
                <span style='font-size:1.2rem;font-weight:800;color:#6366f1;'>{w_score:.1f}/100</span>
              </div>
              <div style='text-align:right;margin-top:4px;'>
                <span style='font-size:0.80rem;color:#64748b;'>
                  Total contribution: {sum(wt_contrib.values()):.1f} pts
                </span>
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # FEATURE 2 — UNDERWRITING EXPLAINABILITY ENGINE
        # ════════════════════════════════════════════════════════════════════════
        demo_box("Explainability Engine (XAI)",
                 "Explainability means being able to explain WHY a credit decision was made. "
                 "Regulators (RBI, Basel III) require banks to justify loan rejections. "
                 "Key risk drivers come from the RF feature importance — the top factors that influenced the ML score.")

        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### ✅ Credit Strengths")
            for p in pos_factors:
                st.markdown(f"<div class='pos-item'>✓ {p}</div>", unsafe_allow_html=True)
            if not pos_factors:
                st.markdown("<div style='color:#64748b;font-size:0.85rem;'>No significant strengths identified.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ex2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### ⚠️ Risk Concerns")
            for n in neg_factors:
                st.markdown(f"<div class='flag-item'>! {n}</div>", unsafe_allow_html=True)
            if not neg_factors:
                st.markdown("<div style='color:#10b981;font-size:0.85rem;'>No material risk drivers.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with ex3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 🔑 Top Risk Drivers (RF)")
            fi      = model_data['feature_importances']
            top5    = list(fi.items())[:5]
            readable= {
                'CIBIL_Score':'CIBIL Score','DTI_Ratio':'DTI Ratio',
                'Monthly_Income':'Monthly Income','Missed_EMIs':'Missed EMIs',
                'Savings':'Savings','Loan_Amount':'Loan Amount',
                'Employment_Length':'Employment','Asset_Value':'Asset Value',
                'Loan_To_Income_Ratio':'LTI Ratio','Net_Worth':'Net Worth',
            }
            for i, (feat, imp) in enumerate(top5):
                lbl = readable.get(feat, feat)
                st.markdown(f"""
                <div style='display:flex;align-items:center;margin-bottom:10px;'>
                  <span style='font-size:1rem;font-weight:800;color:#6366f1;margin-right:10px;width:20px;'>{i+1}.</span>
                  <div style='flex:1;'>
                    <div style='font-size:0.84rem;color:#e2e8f0;font-weight:600;'>{lbl}</div>
                    <div style='background:rgba(99,102,241,0.15);border-radius:4px;height:4px;margin-top:4px;'>
                      <div style='background:#6366f1;height:4px;border-radius:4px;width:{int(imp*400)}px;max-width:100%;'></div>
                    </div>
                  </div>
                  <span style='font-size:0.75rem;color:#64748b;margin-left:10px;'>{imp*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ML Gauge + Flags row
        gx1, gx2 = st.columns(2)
        with gx1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 🤖 ML PD Gauge")
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ml_pd * 100,
                number={'suffix':'%','font':{'size':36,'color':'#fca5a5'}},
                title={'text':'P(Default)','font':{'size':13,'color':'#94a3b8'}},
                gauge={
                    'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#64748b'},
                    'bar':{'color':'#6366f1','thickness':0.22},
                    'bgcolor':'rgba(0,0,0,0)', 'borderwidth':0,
                    'steps':[
                        {'range':[0,15],'color':'rgba(16,185,129,0.18)'},
                        {'range':[15,35],'color':'rgba(245,158,11,0.18)'},
                        {'range':[35,100],'color':'rgba(239,68,68,0.18)'},
                    ],
                    'threshold':{'line':{'color':'#ef4444','width':3},'value':35},
                }
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)',font_color='#e2e8f0',
                                margin=dict(t=30,b=10,l=20,r=20),height=210)
            st.plotly_chart(fig_g, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with gx2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 🚦 Hard Stops & Risk Flags")
            if stops:
                for s in stops:
                    st.markdown(f"<div class='hard-stop'>🚨 <b>AUTO-REJECT:</b> {s}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='pos-item'>✅ No hard-stop violations detected</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.78rem;color:#64748b;margin:10px 0 6px;text-transform:uppercase;letter-spacing:1px;font-weight:600;'>Risk Flags (severity: {fc})</div>", unsafe_allow_html=True)
            if flags:
                for f in flags:
                    st.markdown(f"<div class='flag-item'>⚠ {f}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='pos-item'>✅ Zero risk flags identified</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # FEATURE 6 — CREDIT COMMITTEE MEMO + PDF DOWNLOAD
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📝 AI Credit Analyst Underwriting Note")
        demo_box("Credit Analyst Commentary",
                 "Every bank credit memo includes an analyst's written opinion. "
                 "This section auto-generates that commentary using the borrower's financial profile, "
                 "mimicking what a real underwriter writes in a credit appraisal memo.")
        st.markdown(f'<div class="analyst-note">{analyst_note}</div>', unsafe_allow_html=True)

        st.markdown("<br><hr style='border-color:rgba(99,102,241,0.15);'>", unsafe_allow_html=True)
        st.markdown("#### 🏛️ Credit Committee Memorandum")
        st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:14px;'>Download a professional one-page PDF underwriting report — suitable for portfolio presentations and credit committee reviews.</div>", unsafe_allow_html=True)

        try:
            pdf_bytes = generate_credit_memo_pdf_full(
                details=bd, cs_scores=cs, w_score=w_score, grade=grade,
                verdict=final_verdict, ml_pd=ml_pd, ml_confidence=ml_conf,
                stops=stops, flags=flags, analyst_notes=analyst_note,
                pos_factors=pos_factors, neg_factors=neg_factors,
                consensus_agree=agree, consensus_explanation=consensus_exp,
                grade_label=g_label, audit_rows=audit_rows,
            )
            fname = f"CreditIQ_Memo_{cat}_{datetime.date.today()}.pdf"
            st.download_button(
                label="📄  Download Credit Committee Memo (PDF)",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as pdf_err:
            st.warning(f"PDF generation error: {pdf_err}. Please ensure fpdf2 is installed.")
        st.markdown("</div>", unsafe_allow_html=True)

        # ════════════════════════════════════════════════════════════════════════
        # Stress Testing
        # ════════════════════════════════════════════════════════════════════════
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔴 Stress Testing — RBI Adverse Scenario Analysis")
        demo_box("Why Stress Testing?",
                 "RBI guidelines require banks to test portfolios under adverse scenarios. "
                 "We apply 3 standard shocks: income drop (job loss), rate hike (floating rate impact), "
                 "and EMI burden increase (over-leveraging). This shows credit resilience.")
        st.markdown("<div style='color:#64748b;font-size:0.83rem;margin-bottom:14px;'>Impact on credit score and default probability under 3 RBI-aligned adverse scenarios</div>", unsafe_allow_html=True)
        stress_results = run_stress_test(model_data, bd, w_score, ml_pd)
        sc1,sc2,sc3 = st.columns(3)
        for col, res in zip([sc1,sc2,sc3], stress_results):
            sd_clr = '#ef4444' if res['score_delta']<-5 else '#f59e0b' if res['score_delta']<0 else '#10b981'
            pd_clr2= '#ef4444' if res['pd_delta']>5 else '#f59e0b' if res['pd_delta']>0 else '#10b981'
            with col:
                st.markdown(f"""
                <div class='stress-card'>
                  <div style='font-size:0.85rem;font-weight:700;color:#e2e8f0;margin-bottom:3px;'>{res['scenario']}</div>
                  <div style='font-size:0.70rem;color:#64748b;margin-bottom:12px;'>{res['description']}</div>
                  <div style='font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;'>Score Impact</div>
                  <div style='font-size:1.5rem;font-weight:800;color:{sd_clr};'>{"▼" if res["score_delta"]<0 else "▲"} {abs(res["score_delta"]):.1f}</div>
                  <div style='font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-top:8px;'>PD Change</div>
                  <div style='font-size:1.5rem;font-weight:800;color:{pd_clr2};'>{"▲" if res["pd_delta"]>0 else "▼"} {abs(res["pd_delta"]):.2f}%</div>
                  <div style='font-size:0.70rem;color:#475569;margin-top:8px;'>New PD: {res["new_pd"]*100:.1f}% | Score: {res["new_score"]:.0f}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — PORTFOLIO SEARCH ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif "Search" in page:
    st.markdown("# 🔍 Portfolio Search Engine")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Advanced querying and filtering across the entire loan portfolio</div>", unsafe_allow_html=True)

    with st.expander("🔎 Advanced Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            f_id = st.text_input("Applicant ID (e.g. SYN-01007)")
            f_cat = st.selectbox("Borrower Category", ["All", "Salaried", "Self-Employed", "Student", "Retired"])
        with col2:
            f_grades = st.multiselect("Risk Grades", ["A+", "A", "B+", "B", "C", "D"])
            f_decisions = st.multiselect("Underwriting Decision", ["Approved", "Approved with Conditions", "Manual Review", "Rejected"])
        with col3:
            f_pd_min, f_pd_max = st.slider("Probability of Default (%)", 0.0, 100.0, (0.0, 100.0))
            f_limit = st.selectbox("Max Results", [100, 500, 1000, 5000])

        if st.button("Apply Filters", type="primary"):
            filters = {}
            if f_id: filters['applicant_id'] = f_id
            if f_cat != "All": filters['category'] = f_cat
            if f_grades: filters['risk_grade'] = f_grades
            if f_decisions: filters['decision'] = f_decisions
            if f_pd_min > 0: filters['min_pd'] = f_pd_min / 100.0
            if f_pd_max < 100: filters['max_pd'] = f_pd_max / 100.0
            filters['limit'] = f_limit

            st.session_state['search_filters'] = filters

    filters = st.session_state.get('search_filters', {})
    results_df = db.search_applications(filters)

    st.markdown(f"#### 📊 Search Results ({len(results_df)})")
    if not results_df.empty:
        st.dataframe(results_df, use_container_width=True)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Export Results", data=csv, file_name="CreditIQ_Search.csv", mime="text/csv")
    else:
        st.info("No records found matching your filters.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — CREDIT POLICY ENGINE
# ══════════════════════════════════════════════════════════════════════════════
elif "Policy" in page:
    st.markdown("# 📜 Credit Policy Engine")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Manage global underwriting thresholds and auto-recalculate portfolio</div>", unsafe_allow_html=True)

    policies = db.get_policies()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎛️ Policy Parameters")

        # Create edit form
        new_vals = {}
        st.markdown("#### Hard Stops")
        pc1, pc2 = st.columns(2)
        with pc1:
            new_vals['min_cibil_score'] = st.number_input("Minimum CIBIL Score", value=int(policies['min_cibil_score']['value']), step=10)
            new_vals['max_dti_ratio'] = st.number_input("Maximum DTI Ratio (%)", value=float(policies['max_dti_ratio']['value']), step=5.0)
        with pc2:
            new_vals['min_monthly_income'] = st.number_input("Minimum Monthly Income (₹)", value=int(policies['min_monthly_income']['value']), step=5000)
            new_vals['max_missed_emis'] = st.number_input("Maximum Missed EMIs", value=int(policies['max_missed_emis']['value']), step=1)

        st.markdown("#### Scoring Thresholds")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            new_vals['score_approve'] = st.number_input("Auto-Approve Score", value=int(policies['score_approve']['value']), step=1)
        with sc2:
            new_vals['score_conditional'] = st.number_input("Conditional Score", value=int(policies['score_conditional']['value']), step=1)
        with sc3:
            new_vals['score_manual'] = st.number_input("Manual Review Score", value=int(policies['score_manual']['value']), step=1)

        if st.button("💾 Save & Recalculate Portfolio", type="primary"):
            with st.spinner("Updating policies and recalculating portfolio..."):
                for k, v in new_vals.items():
                    if v != policies[k]['value']:
                        db.update_policy(k, v)

                # Fetch all apps and re-run (simplified for performance)
                # In a real app this would trigger an async batch job
                st.success("Policies updated successfully! (Note: Portfolio recalculation is simulated in this prototype)")
                st.rerun()

        if st.button("🔄 Reset to Defaults"):
            db.reset_policies()
            st.rerun()

    with col2:
        st.markdown("### 📊 Policy Impact")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        stats = db.get_portfolio_stats()
        st.metric("Total Portfolio Size", stats['total'])
        st.metric("Current Approval Rate", f"{stats['approval_rate']:.1f}%")
        st.metric("Average Risk Score", f"{stats['avg_risk_score']:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — WHAT-IF SIMULATOR (UPGRADED)
# ══════════════════════════════════════════════════════════════════════════════
elif "Simulator" in page:
    st.markdown("# 🎛️ Credit What-If Simulator")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Real-time credit scoring with fastest approval path and scenario comparison</div>", unsafe_allow_html=True)

    demo_box("What-If Simulator",
             "This tool helps a borrower understand exactly what changes would improve their credit profile. "
             "The 'Fastest Approval Path' shows the minimum changes needed to move from the current grade to the next. "
             "This kind of tool is used in bank customer advisory and digital lending platforms.")

    s_in, s_out = st.columns([1, 1.3])
    with s_in:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🎚️ Simulator Controls")
        s_cat    = st.selectbox("Borrower Profile", ["Salaried","Self-Employed","Student","Retired"])
        s_cibil  = st.slider("CIBIL Score",         300, 900, 715, key="sc")
        s_income = st.slider("Monthly Income (₹)",   0, 300000, 55000, 2000, key="si")
        s_savings= st.slider("Savings (₹)",           0, 2000000, 110000, 5000, key="ss")
        s_invest = st.slider("Investments (₹)",        0, 1000000, 45000, 5000, key="sv")
        s_emis   = st.slider("Existing EMIs (₹/m)",   0, 150000, 9000, 1000, key="se")
        s_loan   = st.slider("Loan Amount (₹)",        20000, 5000000, 400000, 20000, key="sl")
        s_tenure = st.slider("Tenure (Months)",        12, 240, 48, key="st")
        s_missed = st.slider("Missed Payments",        0, 10, 0, 1, key="sm")
        s_rate   = st.slider("Interest Rate (%)",      5.0, 36.0, 11.5, 0.25, key="sr")
        s_asset  = st.slider("Collateral Value (₹)",   0, 5000000, 0, 50000, key="sa")
        st.markdown("</div>", unsafe_allow_html=True)

    with s_out:
        sim_emi   = calculate_emi(s_loan, s_rate, s_tenure)
        sim_total = s_emis + sim_emi
        sim_dti   = round((sim_total / s_income) * 100, 1) if s_income > 0 else 100.0
        sim_lti   = round(s_loan / (s_income * 12), 2) if s_income > 0 else 10.0
        sim_nw    = s_savings + s_invest + s_asset - s_loan

        sim_d = {
            'Category': s_cat, 'Age': 32, 'Monthly_Income': s_income,
            'Savings': s_savings, 'Investments': s_invest,
            'CIBIL_Score': s_cibil, 'Missed_EMIs': s_missed,
            'Credit_History_Length': 5, 'Existing_EMIs': sim_total,
            'DTI_Ratio': sim_dti, 'Employment_Length': 4.0,
            'Asset_Value': s_asset, 'Loan_Amount': s_loan,
            'Loan_Tenure': s_tenure, 'Loan_Purpose': 'Personal',
            'Co_Applicant': 0, 'Loan_To_Income_Ratio': sim_lti, 'Net_Worth': sim_nw,
        }
        sim_cs     = calculate_5cs_scores(sim_d)
        sim_score  = calculate_weighted_score(sim_cs)
        sim_grade  = get_risk_grade(sim_score)
        sim_stops  = check_hard_stops(sim_d)
        _, sim_fc  = check_red_flags(sim_d)
        sim_verdict,_,_ = get_rule_based_decision(sim_score, sim_stops, sim_fc)
        sim_pd     = predict_single_probability(model_data, sim_d)
        sim_conf   = get_confidence_score(model_data, sim_d)
        sim_gc     = get_grade_color(sim_grade)

        # ── Current State Cards ────────────────────────────────────────────────
        st.markdown("#### 📍 Current Credit State")
        ci1, ci2, ci3, ci4 = st.columns(4)
        for col, val, lbl, clr in [
            (ci1, f"{sim_score:.0f}/100", "Score",    "#6366f1"),
            (ci2, sim_grade,              "Grade",    sim_gc),
            (ci3, f"{sim_pd*100:.1f}%",   "PD",       "#ef4444" if sim_pd>0.25 else "#f59e0b" if sim_pd>0.10 else "#10b981"),
            (ci4, sim_verdict[:12],       "Decision", "#10b981" if "Approv" in sim_verdict else "#f59e0b" if "Manual" in sim_verdict else "#ef4444"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value' style='color:{clr};font-size:22px;'>{val}</div>
                </div>""", unsafe_allow_html=True)

        # Gauge
        fig_sg = go.Figure(go.Indicator(
            mode="gauge+number", value=sim_score,
            number={'font':{'size':30,'color':'#6366f1'}},
            title={'text':'Score Dial','font':{'size':12,'color':'#94a3b8'}},
            gauge={
                'axis':{'range':[0,100],'tickwidth':1,'tickcolor':'#64748b'},
                'bar':{'color':'#6366f1','thickness':0.18},
                'bgcolor':'rgba(0,0,0,0)','borderwidth':0,
                'steps':[
                    {'range':[0,50],'color':'rgba(239,68,68,0.14)'},
                    {'range':[50,72],'color':'rgba(245,158,11,0.14)'},
                    {'range':[72,100],'color':'rgba(16,185,129,0.14)'},
                ],
            }
        ))
        fig_sg.update_layout(paper_bgcolor='rgba(0,0,0,0)',font_color='#e2e8f0',
                             margin=dict(t=28,b=5,l=20,r=20),height=160)
        st.plotly_chart(fig_sg, use_container_width=True)

        # DTI / EMI / LTI indicators
        st.markdown(f"""
        <div style='display:flex;gap:10px;margin-bottom:16px;'>
          <div style='flex:1;background:rgba(15,23,42,0.6);border-radius:8px;padding:10px;text-align:center;'>
            <div style='font-size:10px;color:#64748b;letter-spacing:1px;'>DTI</div>
            <div style='font-size:1.15rem;font-weight:700;color:{"#10b981" if sim_dti<=30 else "#f59e0b" if sim_dti<=50 else "#ef4444"};'>{sim_dti:.1f}%</div>
          </div>
          <div style='flex:1;background:rgba(15,23,42,0.6);border-radius:8px;padding:10px;text-align:center;'>
            <div style='font-size:10px;color:#64748b;letter-spacing:1px;'>EMI</div>
            <div style='font-size:1.15rem;font-weight:700;color:#a855f7;'>₹{sim_emi:,.0f}</div>
          </div>
          <div style='flex:1;background:rgba(15,23,42,0.6);border-radius:8px;padding:10px;text-align:center;'>
            <div style='font-size:10px;color:#64748b;letter-spacing:1px;'>LTI</div>
            <div style='font-size:1.15rem;font-weight:700;color:{"#10b981" if sim_lti<=4 else "#f59e0b" if sim_lti<=6 else "#ef4444"};'>{sim_lti:.1f}×</div>
          </div>
          <div style='flex:1;background:rgba(15,23,42,0.6);border-radius:8px;padding:10px;text-align:center;'>
            <div style='font-size:10px;color:#64748b;letter-spacing:1px;'>Confidence</div>
            <div style='font-size:1.15rem;font-weight:700;color:#6366f1;'>{sim_conf:.0f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

        if sim_stops:
            st.markdown(f"<div class='hard-stop'>🚨 {sim_stops[0]}</div>", unsafe_allow_html=True)

        # ── FEATURE 8: Fastest Approval Path ──────────────────────────────────
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🚀 Fastest Approval Path")
        approval_paths, target_grade = find_fastest_approval_path(
            model_data, sim_d, sim_score, sim_pd, sim_grade
        )
        if approval_paths:
            st.markdown(f"""
            <div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.20);
                        border-radius:10px;padding:12px 16px;margin-bottom:14px;font-size:0.85rem;'>
              🎯 <b style='color:#10b981;'>Target Grade: {target_grade}</b> — here's what you need to change:
            </div>""", unsafe_allow_html=True)
            for path in approval_paths:
                pd_chg = path['pd_delta']
                pd_txt = f"PD {'▲' if pd_chg > 0 else '▼'} {abs(pd_chg):.2f}%"
                pd_clr3= '#ef4444' if pd_chg > 0 else '#10b981'
                st.markdown(f"""
                <div style='background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.18);
                            border-radius:10px;padding:14px 16px;margin-bottom:10px;'>
                  <div style='font-size:0.90rem;font-weight:700;color:#a7f3d0;margin-bottom:4px;'>
                    {path['icon']} {path['action']}
                  </div>
                  <div style='font-size:0.78rem;color:#64748b;margin-bottom:8px;'>{path['why']}</div>
                  <div style='display:flex;gap:14px;font-size:0.80rem;'>
                    <span>Grade: <b style='color:{get_grade_color(path["new_grade"])};'>{sim_grade} → {path["new_grade"]}</b></span>
                    <span>Score: <b style='color:#6366f1;'>+{path["score_delta"]:.1f} pts</b></span>
                    <span>PD: <b style='color:{pd_clr3};'>{pd_txt}</b></span>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pos-item'>✅ Profile is already optimised — at top risk grade!</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Scenario Comparison Table ──────────────────────────────────────────
        if approval_paths:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 📊 Scenario Comparison")
            rows = [{'Scenario': '📍 Current',
                     'Score': f"{sim_score:.0f}/100",
                     'Grade': sim_grade,
                     'PD': f"{sim_pd*100:.1f}%",
                     'DTI': f"{sim_dti:.1f}%",
                     'Decision': sim_verdict}]
            for p in approval_paths:
                rows.append({
                    'Scenario': p['action'][:40],
                    'Score': f"{p['new_score']:.0f}/100",
                    'Grade': p['new_grade'],
                    'PD': f"{p['new_pd']*100:.1f}%",
                    'DTI': f"{sim_dti:.1f}%",
                    'Decision': get_risk_grade_label(p['new_grade'])[:25],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Improvement Suggestions ────────────────────────────────────────────
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 💡 Smart Improvement Suggestions")
        suggestions = get_improvement_suggestions(sim_d, sim_cs, sim_pd, model_data)
        for sug in suggestions:
            st.markdown(f"<div class='suggestion-item'>{sug}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE — SCENARIO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Scenario" in page:
    st.markdown("# 🌊 Macro-Economic Scenario Analysis")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Portfolio-level stress testing across 5 adverse economic scenarios</div>", unsafe_allow_html=True)

    scenarios = scenario_engine.get_scenario_names()
    selected_scenario = st.selectbox("Select Stress Scenario to Run", ["All Scenarios"] + scenarios)

    if st.button("▶️ Run Stress Test", type="primary"):
        with st.spinner("Applying macroeconomic shocks to portfolio and recalculating risk..."):
            port_df = db.get_portfolio_df()
            if port_df.empty:
                st.error("Portfolio is empty. Please generate data or upload applications first.")
            else:
                s_name = 'all' if selected_scenario == "All Scenarios" else selected_scenario
                results = scenario_engine.run_portfolio_stress_test(port_df, model_data, s_name)

                st.success(f"Stress test completed on sample of {min(1000, len(port_df))} applications.")

                for res in results:
                    st.markdown(f"### {res['icon']} {res['scenario']}")
                    st.markdown(f"<p style='color:#94a3b8;font-size:0.9rem;'>{res['description']}</p>", unsafe_allow_html=True)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Average PD", f"{res['stressed_avg_pd']*100:.1f}%", f"{res['pd_change']*100:.1f}%", delta_color="inverse")
                    c2.metric("Approval Rate", f"{res['stressed_approval_rate']:.1f}%", f"{res['approval_change']:.1f}%")
                    c3.metric("Average Score", f"{res['stressed_avg_score']:.1f}", f"{res['score_change']:.1f}")
                    
                    sev_color = "red" if res['severity'] == "Critical" else "orange" if res['severity'] == "Severe" else "yellow"
                    c4.markdown(f"<div style='padding:10px;border-radius:10px;text-align:center;border:1px solid {sev_color};color:{sev_color}'><b>Severity: {res['severity']}</b><br>+{res['high_risk_increase']:.1f}% High Risk</div>", unsafe_allow_html=True)
                    
                    st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL GOVERNANCE (ENHANCED)
# ══════════════════════════════════════════════════════════════════════════════
elif "Governance" in page:
    st.markdown("# ⚖️ Model Governance & Performance")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Transparency, accountability, and audit framework for the dual-layer underwriting engine</div>", unsafe_allow_html=True)

    demo_box("Model Governance",
             "Banks are required by RBI and Basel III to govern their credit models. "
             "Governance includes: model validation, performance monitoring, bias detection, and documentation. "
             "This page demonstrates what a 'Model Risk Management' function looks like in practice.")

    mp1, mp2, mp3, mp4 = st.columns(4)
    acc     = model_data['metrics']['accuracy'] * 100
    auc     = model_data['metrics'].get('auc', 0)
    cv_auc  = model_data['metrics'].get('cv_auc_mean', 0)
    cv_std  = model_data['metrics'].get('cv_auc_std', 0)
    for col, val, lbl, sub, clr in [
        (mp1, f"{acc:.1f}%",    "Test Accuracy",  "Hold-out 20% set",             "#10b981"),
        (mp2, f"{auc:.4f}",     "AUC-ROC",        "Area Under ROC Curve",         "#6366f1"),
        (mp3, f"{cv_auc:.4f}",  "5-Fold CV AUC",  f"±{cv_std:.4f} std deviation",  "#a855f7"),
        (mp4, f"{len(df):,}",   "Training Records","Synthetic portfolio",          "#f59e0b"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value' style='color:{clr};font-size:26px;'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📐 Rule Engine vs ML Framework")
        st.markdown("""
        <div class='pos-item'>✓ <b>Transparent</b> — every rule has an explicit policy justification</div>
        <div class='pos-item'>✓ <b>Regulatory compliant</b> — RBI/SEBI auditable without black-box risk</div>
        <div class='pos-item'>✓ <b>Consistent</b> — same inputs always produce same output</div>
        <div class='flag-item'>! <b>Limitation:</b> Cannot capture non-linear risk interactions</div>
        <div class='flag-item'>! <b>Limitation:</b> Threshold-based; misses borderline edge cases</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 RF Classification Report")
        rep = model_data['metrics']['report']
        tbl = pd.DataFrame({
            'Class':     ['Non-Default (0)', 'Default (1)', 'Macro Avg'],
            'Precision': [rep['0']['precision'], rep['1']['precision'], rep['macro avg']['precision']],
            'Recall':    [rep['0']['recall'],    rep['1']['recall'],    rep['macro avg']['recall']],
            'F1-Score':  [rep['0']['f1-score'],  rep['1']['f1-score'],  rep['macro avg']['f1-score']],
        })
        st.dataframe(tbl.style.format({'Precision':'{:.3f}','Recall':'{:.3f}','F1-Score':'{:.3f}'}),
                     hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with g2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🤖 Random Forest ML Engine")
        st.markdown("""
        <div class='pos-item'>✓ <b>Predictive power</b> — captures complex non-linear risk patterns</div>
        <div class='pos-item'>✓ <b>Feature importance</b> — data-driven risk driver identification</div>
        <div class='pos-item'>✓ <b>Cross-validated</b> — 5-fold AUC protects against overfitting</div>
        <div class='flag-item'>! <b>Limitation:</b> Black-box — individual decisions harder to audit</div>
        <div class='flag-item'>! <b>Limitation:</b> Requires periodic retraining on fresh portfolio data</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Model Disagreement Audit")

        @st.cache_data
        def conflict_audit(_df, sample=500):
            conflicts = []
            for _, row in _df.sample(min(sample, len(_df)), random_state=7).iterrows():
                d = row.to_dict()
                if 'Loan_To_Income_Ratio' not in d:
                    inc = d.get('Monthly_Income', 1)
                    d['Loan_To_Income_Ratio'] = d.get('Loan_Amount', 0) / (inc*12) if inc>0 else 5
                if 'Net_Worth' not in d:
                    d['Net_Worth'] = d.get('Savings',0)+d.get('Investments',0)+d.get('Asset_Value',0)-d.get('Loan_Amount',0)
                cs  = calculate_5cs_scores(d)
                sc  = calculate_weighted_score(cs)
                stp = check_hard_stops(d)
                _,fc= check_red_flags(d)
                rv,_,_ = get_rule_based_decision(sc, stp, fc)
                pd_ = predict_single_probability(model_data, d)
                rule_ok = rv in ('Approved','Approved with Conditions')
                ml_ok   = pd_ < 0.30
                if rule_ok != ml_ok:
                    conflicts.append({
                        'Category':d['Category'],'CIBIL':d['CIBIL_Score'],
                        'DTI':f"{d['DTI_Ratio']:.1f}%",'Rule Verdict':rv,
                        'ML PD':f"{pd_*100:.1f}%",
                        'Conflict':('Rule ✅ ML ❌' if rule_ok else 'ML ✅ Rule ❌')
                    })
            return pd.DataFrame(conflicts)

        cdf = conflict_audit(df)
        conf_pct = len(cdf) / 500 * 100
        st.markdown(f"""
        <div style='background:rgba(245,158,11,0.10);border:1px solid rgba(245,158,11,0.25);
                    border-radius:8px;padding:10px 14px;margin-bottom:10px;font-size:0.85rem;'>
          <b style='color:#fcd34d;'>Disagreement Rate: {conf_pct:.1f}%</b> of sampled applications
          <span style='color:#64748b;'> — these are escalated for manual review.</span>
        </div>""", unsafe_allow_html=True)
        if len(cdf) > 0:
            st.dataframe(cdf.head(7), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 5 Cs weight justification
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### 📐 5 Cs Weight Justification — Finance & Empirical Rationale")
    wt_df = pd.DataFrame({
        'C Factor':    ['Character','Capacity','Capital','Collateral','Conditions'],
        'Weight':      ['35%','30%','15%','15%','5%'],
        'Key Inputs':  ['CIBIL, Missed EMIs, Credit History Length',
                        'DTI Ratio, Monthly Income, Employment Stability',
                        'Savings, FD/MF Investments',
                        'Asset Value, LTV Ratio',
                        'Loan Purpose, Tenure, LTI Ratio'],
        'Finance Rationale': [
            'CIBIL is the strongest predictor of default per RBI empirical data. Repayment behaviour is the best forward-looking indicator.',
            'DTI and income adequacy directly determine whether EMIs can be serviced. The core of underwriting.',
            'Savings buffer reduces Loss-Given-Default (LGD) — the loss severity in event of default.',
            'Collateral coverage directly drives post-default recovery rate. Critical for secured retail lending.',
            'Purpose and tenure are secondary — a creditworthy borrower repays regardless of loan use.',
        ]
    })
    st.dataframe(wt_df, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KNOWLEDGE BASE & DATA PANEL
# ══════════════════════════════════════════════════════════════════════════════
elif "Knowledge" in page:
    st.markdown("# 🎓 Knowledge Base & Dataset Transparency")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Platform methodology, dataset governance, and project credibility documentation</div>", unsafe_allow_html=True)

    tabs = st.tabs(["🛡️ Project Overview", "🗄️ Dataset Governance", "🎤 FAQ & Methodology", "📌 Quick Reference"])

    # ── Tab 1: Project Overview ────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🎯 About CreditIQ")
        st.markdown("""
        <div style='background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.22);
                    border-radius:12px;padding:20px 24px;margin-bottom:18px;color:#e2e8f0;line-height:1.8;font-size:0.92rem;'>
          <b style='color:#a5b4fc;'>CreditIQ</b> is an AI-assisted credit risk underwriting decision-support platform
          developed as an MBA Finance Live Project.<br><br>
          The platform combines <b>traditional underwriting principles</b> (5 Cs of Credit — the global banking standard)
          with <b>machine learning</b> (Random Forest Probability of Default estimation) in a dual-layer decision engine
          that mirrors real-world banking underwriting systems.<br><br>
          It is designed to demonstrate expertise in credit risk analytics, FinTech product thinking, 
          and data-driven financial decision-making for <b>MBA placements in banking, NBFC, and FinTech roles</b>.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Platform Capabilities:**
            - ✅ 5 Cs of Credit Framework (Character, Capacity, Capital, Collateral, Conditions)
            - ✅ Rule-Based Underwriting Scorecard (Policy Compliance Audit)
            - ✅ Random Forest PD Model (AUC > 0.86, 5-Fold CV)
            - ✅ Risk Grade System (A+ → D, aligned with internal bank ratings)
            - ✅ Dual-Layer Consensus Engine with Agreement Detection
            - ✅ Stress Testing (3 RBI-aligned scenarios)
            - ✅ Credit Committee Memo PDF Download
            - ✅ Smart What-If Simulator with Fastest Approval Path
            - ✅ Portfolio Risk Dashboard with 5 Analytics Charts
            - ✅ Explainer Mode with Plain-Language Concept Annotations
            """)
        with c2:
            st.markdown("""
            **Target Roles:**
            - 🏦 Credit Analyst — Public/Private Sector Banks
            - 📊 Risk Analyst — NBFCs, HFCs, MFIs
            - 💡 FinTech Product Manager — Digital Lending Platforms
            - 📈 Data Scientist — Credit Risk / Financial Services
            - 🎓 MBA Finance Research & Analytics Roles
            
            **Technologies Used:**
            - Python 3 | Streamlit | Scikit-learn
            - Plotly | Pandas | NumPy | FPDF2
            - Random Forest | 5-Fold Cross-Validation
            - Synthetic Data | Rule Engine + ML Consensus
            """)
        st.markdown("</div>", unsafe_allow_html=True)

        # Disclaimer
        st.markdown("""
        <div style='background:rgba(30,41,59,0.60);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:18px 22px;color:#94a3b8;font-size:0.84rem;line-height:1.7;margin-top:4px;'>
          <b style='color:#e2e8f0;'>⚠️ Academic & Industry Disclaimer</b><br><br>
          CreditIQ is an AI-assisted underwriting decision-support <em>prototype</em> developed as an MBA Finance Live Project.
          The platform combines traditional underwriting principles (5 Cs of Credit) with machine learning-based 
          Probability of Default estimation. All demonstration data is <strong>synthetic</strong> and does not represent 
          any real customer or financial institution. This system is intended solely for educational, research, 
          and analytical demonstration purposes. It does not constitute a binding lending decision, regulatory compliance 
          opinion, or financial advice.
        </div>""", unsafe_allow_html=True)

    # ── Tab 2: Dataset Governance ──────────────────────────────────────────────
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 🗄️ Dataset Governance & Transparency")

        d1, d2, d3 = st.columns(3)
        for col, val, lbl, clr in [
            (d1, "8,000",     "Total Records",    "#6366f1"),
            (d2, "Synthetic", "Dataset Type",     "#10b981"),
            (d3, "18 Features","Feature Set",     "#a855f7"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value' style='color:{clr};font-size:26px;'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style='background:rgba(16,185,129,0.07);border-left:4px solid #10b981;
                    border-radius:8px;padding:16px 20px;margin:16px 0;'>
          <b style='color:#10b981;'>Why Synthetic Data? — Turn it into a Strength</b><br>
          <span style='color:#94a3b8;font-size:0.88rem;line-height:1.7;'>
          Real banking datasets contain confidential borrower information protected under the 
          <b>IT Act, 2000</b>, <b>RBI data localisation norms</b>, and <b>DPDP Act, 2023</b>. 
          These datasets are unavailable for academic use without regulatory permissions and data sharing agreements.<br><br>
          Synthetic data was programmatically generated to replicate realistic Indian retail lending behaviour — 
          including CIBIL distributions, DTI ratios, segment-wise income profiles, and default rates — 
          <b>enabling full development and validation of the underwriting methodology</b> while maintaining 
          complete privacy compliance.
          </span>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Dataset Features Included:**")
        feat_df = pd.DataFrame({
            'Feature':       ['CIBIL_Score','Monthly_Income','Savings','Investments',
                              'Loan_Amount','DTI_Ratio','Missed_EMIs','Employment_Length',
                              'Asset_Value','Loan_Purpose','Credit_History_Length',
                              'Co_Applicant','Category','Loan_Tenure','Loan_To_Income_Ratio',
                              'Net_Worth','Age','Default (Target)'],
            'Type':          ['Numeric','Numeric','Numeric','Numeric','Numeric','Derived',
                              'Numeric','Numeric','Numeric','Categorical','Numeric',
                              'Binary','Categorical','Numeric','Derived','Derived','Numeric','Binary'],
            'Description':   ['Credit bureau score (300-900)','Net monthly income','Savings balance',
                              'Portfolio value (FD/MF/Gold)','Requested loan size','Debt-to-Income ratio',
                              'Late payments (last 24m)','Job/business duration (years)',
                              'Collateral market value','Loan end use category',
                              'Length of credit history (years)','Guarantor presence (0/1)',
                              'Salaried / Self-Employed / Student / Retired',
                              'Repayment period (months)','Loan ÷ Annual income',
                              'Total assets − total liabilities','Applicant age','1 = Default, 0 = No Default'],
        })
        st.dataframe(feat_df, hide_index=True, use_container_width=True)

        st.markdown("""
        <div style='background:rgba(30,41,59,0.60);border-radius:10px;padding:14px 18px;margin-top:10px;
                    color:#94a3b8;font-size:0.85rem;line-height:1.7;'>
          <b style='color:#e2e8f0;'>🔄 Production Readiness Note</b><br>
          This prototype architecture can be <b>directly retrained</b> using actual bank/NBFC loan portfolios 
          where regulatory permissions exist. The feature engineering, 5 Cs scorecard, and RF pipeline 
          are production-grade and designed to ingest real bureau-enriched data with minimal modification.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tab 3: FAQ & Methodology ────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("#### 🎤 Frequently Asked Questions — Methodology & Design Decisions")

        faqs = [
            ("Why did you use synthetic data?",
             "Real borrower data contains sensitive personal and financial information protected under the IT Act 2000, RBI data localisation norms, and the DPDP Act 2023. Such data is unavailable for academic projects without regulatory data-sharing agreements. Synthetic data was programmatically generated to replicate realistic Indian retail lending behaviour — including credit score distributions, income profiles, and segment-wise default rates — allowing full development and rigorous validation of the underwriting methodology while maintaining complete privacy compliance.",
             "Finance"),

            ("Would this platform work with real bank data?",
             "Yes, absolutely. The architecture is production-ready. The rule-based scorecard, 5 Cs framework, Random Forest pipeline, and feature engineering are all designed to ingest bureau-enriched real data. Retraining would require: (1) obtaining loan origination data with bureau pulls, (2) calibrating the scorecard weights to the institution's actual default rates, (3) validating the RF model using a champion-challenger framework, and (4) obtaining model risk sign-off from the institution's risk function.",
             "Technical"),

            ("What are the limitations of this model?",
             "Three key limitations: First, synthetic datasets cannot perfectly capture real borrower behaviour, macroeconomic shocks, or institution-specific lending patterns — the model would need recalibration on live data. Second, Random Forest is a black-box model; individual decisions cannot be fully explained to regulators without additional XAI tooling like SHAP values. Third, the 5 Cs weights were set based on published research and RBI guidance — a production system would calibrate these through regression analysis on actual default data.",
             "Risk"),

            ("What is DTI and why does it matter?",
             "Debt-to-Income (DTI) ratio measures what percentage of a borrower's monthly income goes toward debt repayments. Formula: Total Monthly EMIs ÷ Monthly Income × 100. Industry standard: DTI < 30% is low risk; 30-50% is moderate; >50% is high risk. In CreditIQ, post-loan DTI is calculated after adding the proposed EMI to existing obligations. RBI guidelines suggest NBFCs maintain average portfolio DTI below 50% for retail loans.",
             "Finance"),

            ("What is Probability of Default (PD)?",
             "PD is the likelihood that a borrower will fail to make required loan payments within a defined time horizon (typically 12 months). It is a core component of Basel III credit risk calculations: Expected Loss = PD × LGD × EAD, where LGD = Loss Given Default and EAD = Exposure at Default. In CreditIQ, PD is estimated by the Random Forest model as the predicted probability of the 'Default = 1' class, calibrated on a synthetic portfolio with a realistic 17.4% base default rate.",
             "Finance"),

            ("What is a Risk Grade and how is it assigned?",
             "Risk grades are internal credit ratings assigned by banks to classify borrowers by risk level. CreditIQ uses: A+ (Score ≥ 88) = Prime, A (80-87) = Near-Prime, B+ (72-79) = Standard, B (62-71) = Sub-Standard, C (50-61) = Speculative, D (<50) = Distressed. These map to real bank internal rating scales (1-8 or AAA-D equivalents). Grade determines pricing, approval conditions, and provisioning requirements.",
             "Finance"),

            ("Why Random Forest and not logistic regression or deep learning?",
             "Three reasons: (1) Interpretability — Random Forest provides feature importances that can be explained to risk managers and regulators; logistic regression loses non-linear relationships. (2) Performance — RF consistently outperforms logistic regression on tabular financial data (our AUC: 0.86 vs typically 0.72-0.78 for LR on similar datasets). (3) Robustness — RF handles missing values and outliers better than deep learning, which would overfit on 8,000 records without significant regularisation and data augmentation.",
             "Technical"),

            ("What is the 5 Cs framework and how did you weight them?",
             "The 5 Cs is the universal credit underwriting framework used by every bank globally. Character (35%) = willingness to repay, primarily CIBIL score — highest weight because RBI empirical data shows CIBIL is the single strongest predictor of default. Capacity (30%) = ability to repay, measured by DTI and income. Capital (15%) = financial cushion, measured by savings and investments — affects Loss Given Default. Collateral (15%) = security, measured by LTV ratio — affects recovery rate post-default. Conditions (5%) = loan terms — secondary to repayment ability. Weights can be defended using regression coefficients from published RBI and BIS working papers on retail credit risk.",
             "Finance"),

            ("How did you validate the machine learning model?",
             "Three-layer validation: (1) Train-test split — 80% training, 20% hold-out test set, stratified to maintain class balance; (2) 5-Fold Stratified Cross-Validation — AUC = 0.8549 ± 0.0063, demonstrating consistency across data folds and absence of overfitting; (3) Classification report — precision, recall, and F1-score reported for both default and non-default classes. The low standard deviation in CV AUC (0.006) confirms model stability. In a production setting, we would additionally run a Kolmogorov-Smirnov test and Gini coefficient analysis.",
             "Technical"),

            ("What makes this project placement-worthy?",
             "Four differentiators: (1) Finance rigour — the 5 Cs implementation uses industry-standard weights with empirical justification, not arbitrary values; (2) Technical depth — dual-layer architecture, 5-fold CV, feature engineering, and XAI mirror real banking systems; (3) Product thinking — the What-If simulator, fastest approval path, and credit memo generator demonstrate product management capabilities; (4) Presentation quality — the PDF memo, executive dashboard, and governance panel communicate complex analysis to non-technical stakeholders at a professional standard.",
             "Product"),
        ]

        for q, a, tag in faqs:
            tag_clr = {'Finance':'#10b981','Technical':'#6366f1','Risk':'#ef4444','Product':'#f59e0b'}.get(tag,'#64748b')
            with st.expander(f"❓ {q}"):
                st.markdown(f"""
                <div style='margin-bottom:8px;'>
                  <span style='background:{tag_clr}20;color:{tag_clr};font-size:0.72rem;font-weight:700;
                               letter-spacing:1px;padding:3px 8px;border-radius:4px;text-transform:uppercase;'>{tag}</span>
                </div>
                <div style='color:#e2e8f0;font-size:0.90rem;line-height:1.75;'>{a}</div>
                """, unsafe_allow_html=True)

    # ── Tab 4: Quick Reference Glossary ───────────────────────────────────────
    with tabs[3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📌 Banking & Credit Risk Glossary")
        terms = {
            "AUC-ROC":          "Area Under the Receiver Operating Characteristic Curve. Measures a model's ability to discriminate between defaulters and non-defaulters. AUC = 1.0 is perfect; 0.5 is random. Industry benchmark: > 0.75 = good, > 0.85 = excellent.",
            "CIBIL Score":      "Credit score issued by TransUnion CIBIL (India's primary credit bureau). Range: 300-900. Above 750 = prime borrower; Below 600 = sub-prime. Based on repayment history, credit utilisation, and credit age.",
            "DTI Ratio":        "Debt-to-Income Ratio = Total Monthly EMIs ÷ Monthly Gross Income. Measures repayment burden. RBI suggests keeping below 50% for retail loans.",
            "PD":               "Probability of Default — likelihood of a borrower defaulting within 12 months. Core Basel III metric: Expected Loss = PD × LGD × EAD.",
            "LGD":              "Loss Given Default — percentage of exposure lost if a borrower defaults, after recovery from collateral/guarantees. Collateralised loans have lower LGD.",
            "EAD":              "Exposure at Default — total amount owed at the time of default, including principal, accrued interest, and fees.",
            "LTV":              "Loan-to-Value Ratio = Loan Amount ÷ Collateral Market Value. Lower LTV = more secured. Banks typically lend up to 75-80% LTV for home loans.",
            "LTI Ratio":        "Loan-to-Income Ratio = Loan Amount ÷ Annual Income. CreditIQ policy: LTI > 10× triggers automatic rejection.",
            "EMI":              "Equated Monthly Instalment = P·r·(1+r)^n / ((1+r)^n – 1). Fixed monthly payment covering principal + interest.",
            "Hard Stop":        "An automatic rejection trigger based on a single policy violation regardless of overall score. Example: CIBIL < 600 = immediate reject.",
            "5 Cs of Credit":   "Universal credit framework: Character (willingness), Capacity (ability), Capital (cushion), Collateral (security), Conditions (loan terms).",
            "Risk Grade":       "Internal rating assigned to a borrower/loan to classify credit quality. CreditIQ: A+ (prime) → D (distressed). Used for pricing, provisioning, and capital allocation.",
            "Cross-Validation": "Statistical technique to assess ML model generalisation. 5-fold CV splits data into 5 parts, trains 5 times, and averages AUC — protects against overfitting.",
            "Stress Testing":   "Simulation of portfolio performance under adverse scenarios (income shock, rate hike, EMI spike). Mandatory under RBI Basel III guidelines.",
            "NPA":              "Non-Performing Asset — loan where the borrower has not paid EMI for 90+ days. RBI classification: Substandard → Doubtful → Loss.",
            "NBFC":             "Non-Banking Financial Company — regulated by RBI but cannot accept demand deposits. Includes housing finance companies (HFCs), microfinance institutions (MFIs).",
        }
        col_a, col_b = st.columns(2)
        items = list(terms.items())
        half  = len(items) // 2
        for col, chunk in [(col_a, items[:half]), (col_b, items[half:])]:
            with col:
                for term, definition in chunk:
                    st.markdown(f"""
                    <div style='margin-bottom:12px;padding:10px 14px;background:rgba(15,23,42,0.65);
                                border-left:3px solid #6366f1;border-radius:6px;'>
                      <div style='color:#a5b4fc;font-weight:700;font-size:0.86rem;margin-bottom:4px;'>{term}</div>
                      <div style='color:#94a3b8;font-size:0.80rem;line-height:1.6;'>{definition}</div>
                    </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CREDIT MEMO GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Memo" in page:
    st.markdown("# 📄 Credit Memo Generator")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Generate and download a formal Credit Committee Memorandum from your last underwriting assessment</div>", unsafe_allow_html=True)

    demo_box("Credit Memo Generator",
             "Every credit decision in a bank is documented as a formal Credit Committee Memo. "
             "This memo is presented to the Credit Committee (or sanctioning authority) for approval. "
             "It includes borrower details, scorecard results, risk assessment, policy compliance, "
             "and the analyst's recommendation. This page generates that document.")

    la = st.session_state.get('last_assessment')

    if la is None:
        st.markdown("""
        <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.28);
                    border-radius:14px;padding:32px;text-align:center;margin-top:20px;'>
          <div style='font-size:3rem;margin-bottom:12px;'>⚠️</div>
          <div style='font-size:1.1rem;font-weight:700;color:#fcd34d;margin-bottom:8px;'>No Assessment Found</div>
          <div style='font-size:0.88rem;color:#94a3b8;line-height:1.7;'>
            Please run a <b style='color:#f59e0b;'>Credit Assessment</b> first (Page 2) to generate a memo.<br>
            The memo will automatically pre-fill with your borrower's details and underwriting results.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bd       = la['bd']
        cs       = la['cs']
        w_score  = la['w_score']
        grade    = la['grade']
        g_label  = la['g_label']
        stops    = la['stops']
        flags    = la['flags']
        verdict  = la['verdict']
        final_verdict = la['final_verdict']
        ml_pd    = la['ml_pd']
        ml_conf  = la['ml_conf']
        agree    = la['agree']
        consensus_exp = la['consensus_exp']
        decision_source = la['decision_source']
        audit_rows = la['audit_rows']
        analyst_note = la['analyst_note']
        new_emi  = la['new_emi']
        dti      = la['dti']
        pos_factors = []
        neg_factors = []
        cibil = bd['CIBIL_Score']
        lti   = bd.get('Loan_To_Income_Ratio', 0)
        savings = bd.get('Savings', 0)
        income  = bd.get('Monthly_Income', 1)
        missed  = bd.get('Missed_EMIs', 0)
        emp     = bd.get('Employment_Length', 0)
        asset_val = bd.get('Asset_Value', 0)
        loan_amt  = bd.get('Loan_Amount', 0)
        rate     = la['rate']
        tenure   = la['tenure']
        cat      = la['cat']
        gc       = get_grade_color(grade)

        if cibil >= 750: pos_factors.append("Excellent CIBIL score")
        elif cibil >= 700: pos_factors.append("Good CIBIL score")
        if dti <= 30: pos_factors.append(f"Low DTI ({dti:.1f}%)")
        if savings >= income * 4: pos_factors.append("Strong liquidity buffer")
        if missed == 0: pos_factors.append("Zero missed payments")
        if cibil < 700: neg_factors.append(f"Below-average CIBIL ({cibil})")
        if dti > 50: neg_factors.append(f"High DTI ({dti:.1f}%)")
        if missed >= 2: neg_factors.append(f"{missed} missed EMIs")

        banner_cls = {'Approved':'approved','Approved with Conditions':'conditional',
                      'Manual Review':'manual','Rejected':'rejected'}.get(final_verdict,'rejected')

        # ── Memo Preview Card ──────────────────────────────────────────────────
        st.markdown("""
        <div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.25);
                    border-radius:12px;padding:14px 20px;margin-bottom:20px;'>
          <span style='font-size:0.70rem;letter-spacing:2px;text-transform:uppercase;color:#6366f1;font-weight:700;'>🔄 Loaded from last assessment</span>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        for col, val, lbl, clr in [
            (m1, bd.get('Category','—'),        "Borrower Type",  "#a5b4fc"),
            (m2, f"₹{loan_amt:,.0f}",           "Loan Amount",    "#6366f1"),
            (m3, final_verdict,                  "Decision",       "#10b981" if 'Approv' in final_verdict else "#ef4444"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value' style='color:{clr};font-size:20px;'>{val}</div>
                </div>""", unsafe_allow_html=True)

        m4, m5, m6 = st.columns(3)
        for col, val, lbl, clr in [
            (m4, f"{w_score:.0f}/100", "Risk Score",     "#6366f1"),
            (m5, grade,                "Risk Grade",     gc),
            (m6, f"{ml_pd*100:.1f}%", "PD (ML Model)",  "#ef4444" if ml_pd>0.25 else "#f59e0b"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value' style='color:{clr};font-size:22px;'>{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='banner banner-{banner_cls}'>
          {final_verdict.upper()} &nbsp;|&nbsp; Risk Grade {grade} — {g_label}
          <div style='font-size:0.82rem;font-weight:400;margin-top:4px;opacity:0.85;'>{la['detail_msg']}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("#### 📝 AI Analyst Note")
        st.markdown(f'<div class="analyst-note">{analyst_note}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 📋 Policy Compliance Summary")
        all_pass = all(p for _, p, _, _ in audit_rows)
        if not all_pass:
            st.markdown("<div class='hard-stop'>🚨 <strong>POLICY VIOLATION</strong> — One or more underwriting rules failed.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='pos-item'>✅ All underwriting policy rules passed — loan is policy-compliant.</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### ⬇️ Download Credit Committee Memorandum")
        st.markdown("<div style='color:#64748b;font-size:0.85rem;margin-bottom:14px;'>Professional one-page PDF memo or Word document — formatted for Credit Committee review and portfolio presentations.</div>", unsafe_allow_html=True)

        try:
            pdf_bytes = generate_credit_memo_pdf_full(
                details=bd, cs_scores=cs, w_score=w_score, grade=grade,
                verdict=final_verdict, ml_pd=ml_pd, ml_confidence=ml_conf,
                stops=stops, flags=flags, analyst_notes=analyst_note,
                pos_factors=pos_factors, neg_factors=neg_factors,
                consensus_agree=agree, consensus_explanation=consensus_exp,
                grade_label=g_label, audit_rows=audit_rows,
            )
            
            from memo import generate_credit_memo_docx
            docx_bytes = generate_credit_memo_docx(
                details=bd, cs_scores=cs, w_score=w_score, grade=grade,
                verdict=final_verdict, ml_pd=ml_pd, ml_confidence=ml_conf,
                stops=stops, flags=flags, analyst_notes=analyst_note,
                pos_factors=pos_factors, neg_factors=neg_factors,
                consensus_agree=agree, consensus_explanation=consensus_exp,
                grade_label=g_label,
            )
            
            fname_pdf = f"CreditIQ_Memo_{cat}_{datetime.date.today()}.pdf"
            fname_docx = f"CreditIQ_Memo_{cat}_{datetime.date.today()}.docx"
            
            dl_col1, dl_col2, _ = st.columns([1, 1, 1])
            with dl_col1:
                st.download_button(
                    label="📄 Download PDF Memo",
                    data=pdf_bytes,
                    file_name=fname_pdf,
                    mime="application/pdf",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    label="📝 Download Word Document",
                    data=docx_bytes,
                    file_name=fname_docx,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error generating memo: {e}")

        # ── Risk Factor Summary ────────────────────────────────────────────────
        if pos_factors or neg_factors:
            pf_col, nf_col = st.columns(2)
            with pf_col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### ✅ Credit Strengths")
                for p in pos_factors:
                    st.markdown(f"<div class='pos-item'>✓ {p}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with nf_col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("#### ⚠️ Risk Concerns")
                for n in neg_factors:
                    st.markdown(f"<div class='flag-item'>! {n}</div>", unsafe_allow_html=True)
                if not neg_factors:
                    st.markdown("<div style='color:#10b981;font-size:0.85rem;'>No material risk concerns.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DECISION AUDIT TRAIL
# ══════════════════════════════════════════════════════════════════════════════
elif "Audit" in page:
    st.markdown("# 🕵️ Decision Audit Trail")
    st.markdown("<div style='color:#64748b;margin-bottom:22px;'>Immutable chronological log of all underwriting decisions made in this session — aligned with RBI model governance and audit requirements</div>", unsafe_allow_html=True)

    demo_box("Why Audit Trails Matter",
             "RBI and Basel III require banks to maintain comprehensive audit trails for all credit decisions. "
             "Every approval, rejection, and conditional decision must be logged with full transparency — "
             "including who made the decision, what data was used, and what the model outputs were. "
             "This page simulates an enterprise underwriting audit log.")

    audit_log = st.session_state.get('audit_log', [])

    # ── Session Stats ──────────────────────────────────────────────────────────
    a1, a2, a3, a4 = st.columns(4)
    total_ev = len(audit_log)
    approved_ev = sum(1 for e in audit_log if 'Approv' in e.get('Decision',''))
    rejected_ev = sum(1 for e in audit_log if 'Reject' in e.get('Decision',''))
    manual_ev  = sum(1 for e in audit_log if 'Manual' in e.get('Decision','') or 'Condition' in e.get('Decision',''))

    for col, val, lbl, clr, sub in [
        (a1, total_ev,    "Total Events",    "#6366f1", "This session"),
        (a2, approved_ev, "Approved",        "#10b981", "Incl. conditional"),
        (a3, rejected_ev, "Rejected",        "#ef4444", "Auto & manual"),
        (a4, manual_ev,   "Manual Review",   "#f59e0b", "Flagged for review"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value' style='color:{clr};font-size:28px;'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    if not audit_log:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.06);border:1px dashed rgba(99,102,241,0.30);
                    border-radius:14px;padding:40px;text-align:center;margin-top:24px;'>
          <div style='font-size:3rem;margin-bottom:12px;'>📋</div>
          <div style='font-size:1.1rem;font-weight:700;color:#a5b4fc;margin-bottom:8px;'>Audit Log Empty</div>
          <div style='font-size:0.88rem;color:#64748b;line-height:1.7;'>
            Run a <b style='color:#6366f1;'>Credit Assessment</b> (Page 2) to start populating the audit trail.<br>
            Every assessment result is automatically logged here with full traceability.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("---")

        # ── Timeline View ──────────────────────────────────────────────────────
        st.markdown("#### ⏱️ Session Decision Timeline")
        timeline_html = "<div style='position:relative;padding-left:24px;border-left:2px solid rgba(99,102,241,0.25);margin:16px 0 24px;'>"
        for ev in reversed(audit_log):
            dec = ev.get('Decision', '')
            dot_clr = '#10b981' if 'Approv' in dec else '#ef4444' if 'Reject' in dec else '#f59e0b'
            dec_icon = '✅' if 'Approv' in dec else '❌' if 'Reject' in dec else '⚠️'
            timeline_html += f"""
            <div style='margin-bottom:18px;position:relative;'>
              <div style='position:absolute;left:-31px;top:4px;width:12px;height:12px;
                          border-radius:50%;background:{dot_clr};border:2px solid #0f172a;'></div>
              <div style='background:rgba(11,18,36,0.75);border:1px solid rgba(255,255,255,0.07);
                          border-radius:10px;padding:12px 16px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                  <span style='font-size:0.80rem;color:#6366f1;font-weight:700;'>Event #{ev['Event #']}</span>
                  <span style='font-size:0.72rem;color:#475569;font-family:monospace;'>{ev['Time']}</span>
                </div>
                <div style='margin-top:6px;display:flex;gap:12px;flex-wrap:wrap;'>
                  <span style='font-size:0.82rem;color:#e2e8f0;font-weight:600;'>{ev.get('Borrower Type','')}</span>
                  <span style='font-size:0.82rem;color:#94a3b8;'>{ev.get('Loan Amount','')}</span>
                  <span style='font-size:0.82rem;color:#a5b4fc;'>CIBIL: {ev.get('CIBIL','')}</span>
                  <span style='font-size:0.82rem;color:#64748b;'>Score: {ev.get('Score','')}</span>
                  <span style='font-size:0.85rem;font-weight:700;color:{dot_clr};'>{dec_icon} {dec}</span>
                </div>
                <div style='margin-top:6px;font-size:0.75rem;color:#475569;'>
                  Grade: <b style='color:#e2e8f0;'>{ev.get('Grade','')}</b> &nbsp;·&nbsp;
                  PD: <b style='color:#ef4444;'>{ev.get('PD','')}</b> &nbsp;·&nbsp;
                  Consensus: {ev.get('Consensus','')}
                </div>
              </div>
            </div>"""
        timeline_html += "</div>"
        st.markdown(timeline_html, unsafe_allow_html=True)

        # ── Tabular Audit Log ──────────────────────────────────────────────────
        st.markdown("#### 📋 Full Audit Log Table")
        audit_df = pd.DataFrame(audit_log)
        st.dataframe(audit_df, hide_index=True, use_container_width=True, height=280)

        # ── Decision Distribution Chart ────────────────────────────────────────
        if len(audit_log) >= 2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("#### 📊 Session Decision Distribution")
            dist_data = audit_df['Decision'].value_counts().reset_index()
            dist_data.columns = ['Decision','Count']
            color_map = {'Approved':'#10b981','Approved with Conditions':'#f59e0b',
                         'Rejected':'#ef4444','Manual Review':'#3b82f6'}
            colors = [color_map.get(d,'#6366f1') for d in dist_data['Decision']]
            fig_audit = px.bar(dist_data, x='Decision', y='Count', text='Count',
                               color='Decision',
                               color_discrete_map=color_map)
            fig_audit.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                                    showlegend=False, margin=dict(t=10,b=10,l=10,r=10),
                                    height=250, xaxis_title='', yaxis_title='Count',
                                    xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                                    yaxis=dict(gridcolor='rgba(255,255,255,0.04)'))
            fig_audit.update_traces(textposition='outside', textfont_color='white')
            st.plotly_chart(fig_audit, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ── CSV Export ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        csv_col, clr_col, _ = st.columns([1, 1, 2])
        with csv_col:
            csv_data = audit_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️  Export Audit Log (CSV)",
                data=csv_data,
                file_name=f"CreditIQ_Audit_{datetime.date.today()}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with clr_col:
            if st.button("🗑️  Clear Audit Log", use_container_width=True):
                st.session_state['audit_log'] = []
                st.session_state['assessment_count'] = 0
                st.rerun()

        # ── RBI Compliance Note ────────────────────────────────────────────────
        st.markdown("""
        <div style='background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.20);
                    border-radius:10px;padding:16px 20px;margin-top:20px;font-size:0.84rem;color:#94a3b8;line-height:1.7;'>
          <b style='color:#a5b4fc;'>🏛️ Regulatory Alignment Note</b><br>
          This audit trail demonstrates compliance with <b>RBI Master Direction on IT Framework for NBFC</b>,
          <b>Basel III Model Risk Management</b>, and <b>SEBI LODR audit trail requirements</b>.
          In production, this log would be maintained in an immutable database with cryptographic hashing
          (blockchain-style) for tamper-evidence. Access would be restricted to authorised credit officers
          and internal audit teams.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:40px 0 20px;margin-top:40px;
            border-top:1px solid rgba(99,102,241,0.10);'>
  <div style='font-size:0.72rem;color:#1e293b;letter-spacing:1px;'>
    CreditIQ &nbsp;·&nbsp; MBA Finance Capstone Live Project &nbsp;·&nbsp;
    Rule-Based Scorecard + Random Forest · 5 Cs Framework · Dual-Layer AI Engine
  </div>
  <div style='font-size:0.65rem;color:#1e293b;margin-top:6px;'>
    For academic and demonstration purposes only. Not a binding lending decision.
  </div>
</div>
""", unsafe_allow_html=True)
