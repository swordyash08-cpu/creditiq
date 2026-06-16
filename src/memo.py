"""
CreditIQ — Credit Committee Memorandum Generator
Produces a professional A4 PDF using fpdf2 (pip install fpdf2).
All rupee symbols replaced with 'Rs.' for PDF font compatibility.
"""
from fpdf import FPDF
import datetime
import re


def _clean(text: str) -> str:
    """Strip/replace characters unsupported by PDF built-in fonts."""
    replacements = {
        '₹': 'Rs.', '→': '->', '•': '*', '✓': '[OK]', '✅': '[OK]',
        '⚠️': '[!]', '⚠': '[!]', '❌': '[X]', '🎯': '>',
        '💰': '>', '🏦': '>', '🔄': '>', '📉': 'v', '📈': '^',
        '💳': '>', '×': 'x', '≥': '>=', '≤': '<=',
        '\u2013': '-', '\u2014': '--', '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u00b1': '+/-',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Remove any remaining non-latin1 characters
    return text.encode('latin-1', errors='replace').decode('latin-1')


class _MemoPDF(FPDF):
    def __init__(self, ref_id):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_margins(20, 15, 20)
        self.set_auto_page_break(auto=True, margin=22)
        self._ref = ref_id

    def header(self):
        # Dark header band
        self.set_fill_color(15, 23, 42)
        self.rect(0, 0, 210, 22, 'F')
        self.set_xy(20, 5)
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(255, 255, 255)
        self.cell(85, 7, 'CreditIQ', new_x='RIGHT')
        self.set_font('Helvetica', '', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 7, f'Ref: {self._ref}', align='R', new_x='LMARGIN', new_y='NEXT')
        self.set_x(20)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 5, 'AI-Assisted Credit Risk Underwriting Platform', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def footer(self):
        self.set_y(-14)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(100, 116, 139)
        self.cell(0, 4,
                  'CONFIDENTIAL | MBA Finance Live Project | CreditIQ Prototype | Data is synthetic — not real customer data.',
                  align='C', new_x='LMARGIN', new_y='NEXT')
        self.cell(0, 4,
                  f'Page {self.page_no()} | Generated {datetime.datetime.now().strftime("%d %b %Y, %H:%M")}',
                  align='C')

    def section_header(self, title: str):
        self.set_fill_color(79, 70, 229)          # indigo-600
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 9)
        self.cell(0, 6, f'  {_clean(title)}', fill=True, new_x='LMARGIN', new_y='NEXT')
        self.set_text_color(30, 41, 59)
        self.ln(2)

    def kv_row(self, label: str, value: str, label_w: int = 65):
        self.set_font('Helvetica', '', 8)
        self.set_text_color(100, 116, 139)
        self.cell(label_w, 5, _clean(label))
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(30, 41, 59)
        self.cell(0, 5, _clean(str(value)), new_x='LMARGIN', new_y='NEXT')


def generate_credit_memo_pdf(
    details, cs_scores, w_score, grade, verdict,
    ml_pd, ml_confidence, stops, flags,
    analyst_notes, pos_factors, neg_factors,
    consensus_agree, consensus_explanation, grade_label='',
) -> bytes:
    """
    Generates a professional A4 Credit Committee Memorandum.
    Returns PDF content as bytes (ready for st.download_button).
    """
    today  = datetime.date.today()
    ref_id = f"CIQ-{today.strftime('%Y%m%d')}-{abs(hash(str(details))) % 9999:04d}"

    # Recommendation colour
    if verdict == 'Approved':
        rec_rgb = (16, 185, 129)    # green
    elif verdict in ('Approved with Conditions', 'Manual Review'):
        rec_rgb = (217, 119, 6)     # amber
    else:
        rec_rgb = (220, 38, 38)     # red

    pdf = _MemoPDF(ref_id)
    pdf.add_page()

    # ── DOCUMENT TITLE ─────────────────────────────────────────────────────────
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, 'CREDIT COMMITTEE MEMORANDUM', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5,
             f'Date: {today.strftime("%d %B %Y")}  |  Prepared by: AI Underwriting Engine  |  Status: DRAFT',
             align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    # ── EXECUTIVE SUMMARY BOX ──────────────────────────────────────────────────
    box_y = pdf.get_y()
    pdf.set_fill_color(245, 247, 252)
    pdf.set_draw_color(99, 102, 241)
    pdf.rect(20, box_y, 170, 24, 'FD')

    col_w   = 42
    labels  = ['Risk Score', 'Risk Grade', 'Default Prob.', 'Decision']
    values  = [f'{w_score:.0f}/100', grade, f'{ml_pd*100:.1f}%', verdict[:18]]
    pd_clr  = (16, 185, 129) if ml_pd < 0.15 else (217, 119, 6) if ml_pd < 0.30 else (220, 38, 38)
    val_clrs = [(79, 70, 229), (79, 70, 229), pd_clr, rec_rgb]

    pdf.set_y(box_y + 3)
    pdf.set_x(20)
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(100, 116, 139)
    for lbl in labels:
        pdf.cell(col_w, 4, lbl, align='C')
    pdf.ln(4)
    pdf.set_x(20)
    for val, clr in zip(values, val_clrs):
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(*clr)
        pdf.cell(col_w, 9, _clean(val), align='C')
    pdf.ln(12)
    pdf.ln(2)

    # ── SECTION 1: APPLICANT PROFILE ──────────────────────────────────────────
    pdf.section_header('1.  APPLICANT PROFILE')
    cat_labels = {
        'Salaried': 'Salaried Professional', 'Self-Employed': 'Self-Employed Business Owner',
        'Student': 'Student Borrower', 'Retired': 'Retired Individual',
    }
    rows1 = [
        ('Borrower Category',  cat_labels.get(details.get('Category', ''), details.get('Category', ''))),
        ('Age',                f"{details.get('Age', '-')} years"),
        ('Monthly Income',     f"Rs. {details.get('Monthly_Income', 0):,.0f}"),
        ('Loan Amount',        f"Rs. {details.get('Loan_Amount', 0):,.0f}"),
        ('Loan Purpose',       details.get('Loan_Purpose', '-')),
        ('Loan Tenure',        f"{details.get('Loan_Tenure', 0)} months"),
        ('Co-Applicant',       'Yes' if details.get('Co_Applicant', 0) else 'No'),
        ('CIBIL Score',        str(details.get('CIBIL_Score', '-'))),
    ]
    for label, val in rows1:
        pdf.kv_row(label, val)
    pdf.ln(3)

    # ── SECTION 2: CREDIT RISK ASSESSMENT ─────────────────────────────────────
    pdf.section_header('2.  CREDIT RISK ASSESSMENT')
    rows2 = [
        ('Rule-Based Score (5 Cs)', f'{w_score:.1f} / 100'),
        ('Risk Grade',              f'{grade} — {grade_label}'),
        ('ML Probability of Default', f'{ml_pd*100:.1f}%'),
        ('ML Model Confidence',     f'{ml_confidence:.1f}%'),
        ('DTI Ratio',               f"{details.get('DTI_Ratio', 0):.1f}%"),
        ('Missed EMIs (24 months)', str(details.get('Missed_EMIs', '-'))),
        ('Loan-to-Income Ratio',    f"{details.get('Loan_To_Income_Ratio', 0):.1f}x"),
        ('Model Consensus',         'AGREE' if consensus_agree else 'DISAGREE — Manual Review Required'),
    ]
    for label, val in rows2:
        pdf.kv_row(label, val)

    if not consensus_agree:
        pdf.ln(1)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(180, 100, 0)
        pdf.multi_cell(0, 4, _clean(f'Note: {consensus_explanation}'))
    pdf.ln(3)

    # ── SECTION 3: 5 Cs SCORECARD ─────────────────────────────────────────────
    pdf.section_header('3.  FIVE Cs CREDIT SCORECARD')
    weight_map  = {'Character': 35, 'Capacity': 30, 'Capital': 15, 'Collateral': 15, 'Conditions': 5}

    for c_name, c_val in cs_scores.items():
        w_pct        = weight_map.get(c_name, 0)
        contribution = round(c_val * (w_pct / 100) * 10, 1)
        bar_total    = 90      # mm
        filled_mm    = int(c_val / 10 * bar_total)

        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(28, 5, c_name)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(10, 5, f'{w_pct}%', align='C')

        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1.5

        # Background bar
        pdf.set_fill_color(220, 222, 240)
        pdf.rect(bar_x, bar_y, bar_total, 3, 'F')
        # Filled bar
        if c_val >= 7:
            pdf.set_fill_color(16, 185, 129)
        elif c_val >= 5:
            pdf.set_fill_color(217, 119, 6)
        else:
            pdf.set_fill_color(220, 38, 38)
        if filled_mm > 0:
            pdf.rect(bar_x, bar_y, filled_mm, 3, 'F')

        pdf.set_x(bar_x + bar_total + 2)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(79, 70, 229)
        pdf.cell(14, 5, f'{c_val:.1f}/10')
        pdf.set_font('Helvetica', '', 7)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, f'(+{contribution} pts)', new_x='LMARGIN', new_y='NEXT')

    pdf.ln(3)

    # ── SECTION 4: POLICY COMPLIANCE AUDIT ────────────────────────────────────
    pdf.section_header('4.  UNDERWRITING POLICY COMPLIANCE AUDIT')

    # Table header
    pdf.set_fill_color(241, 245, 249)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(78, 6, 'Policy Rule',   fill=True, border=1)
    pdf.cell(38, 6, 'Actual Value',  fill=True, border=1, align='C')
    pdf.cell(26, 6, 'Status',        fill=True, border=1, align='C')
    pdf.cell(28, 6, 'Pass?',         fill=True, border=1, align='C', new_x='LMARGIN', new_y='NEXT')

    # We need to import from model in the context of memo being called from app
    # Caller will pass audit results directly (see generate_credit_memo_pdf signature below)
    # Actually, let's accept audit as parameter
    pass  # handled after function via audit_rows parameter

    pdf.ln(3)

    # ── SECTION 5: STRENGTHS & CONCERNS ───────────────────────────────────────
    pdf.section_header('5.  KEY STRENGTHS & RISK CONCERNS')

    x_l, x_r, col_w2 = 20, 107, 84

    # Headers
    pdf.set_xy(x_l, pdf.get_y())
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(16, 150, 100)
    pdf.cell(col_w2, 5, 'KEY STRENGTHS')
    pdf.set_xy(x_r, pdf.get_y())
    pdf.set_text_color(200, 50, 50)
    pdf.cell(col_w2, 5, 'RISK CONCERNS')
    pdf.ln(6)

    n = max(len(pos_factors[:5]), len(neg_factors[:5]))
    for i in range(n):
        row_y = pdf.get_y()
        next_y = row_y

        if i < len(pos_factors):
            pdf.set_xy(x_l, row_y)
            pdf.set_font('Helvetica', '', 7)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(col_w2 - 2, 4, _clean(f'+ {pos_factors[i]}'))
            next_y = max(next_y, pdf.get_y())

        if i < len(neg_factors):
            pdf.set_xy(x_r, row_y)
            pdf.set_font('Helvetica', '', 7)
            pdf.set_text_color(30, 41, 59)
            pdf.multi_cell(col_w2 - 2, 4, _clean(f'- {neg_factors[i]}'))
            next_y = max(next_y, pdf.get_y())

        pdf.set_y(next_y + 1)

    pdf.ln(3)

    # ── SECTION 6: ANALYST COMMENTARY ─────────────────────────────────────────
    pdf.section_header('6.  CREDIT ANALYST COMMENTARY')
    pdf.set_font('Helvetica', 'I', 8.5)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 4.5, _clean(analyst_notes))
    pdf.ln(3)

    # ── SECTION 7: RECOMMENDATION ─────────────────────────────────────────────
    pdf.section_header('7.  FINAL RECOMMENDATION')
    pdf.set_fill_color(*rec_rgb)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 13)
    pdf.cell(0, 11, f'  {verdict.upper()}', fill=True, new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    cond_text = {
        'Approved':
            'Standard approval at the best applicable interest rate. No additional conditions required. '
            'Proceed with loan documentation and disbursement.',
        'Approved with Conditions':
            'Approval subject to: (a) enhanced interest rate margin of 1-2% above base rate, '
            '(b) co-applicant or additional collateral if DTI exceeds 45%, '
            '(c) reduced loan quantum or staged disbursement, '
            '(d) periodic income review every 12 months.',
        'Manual Review':
            'Borderline case requiring senior underwriter review. Actions: '
            '(a) collect supplementary income documentation, '
            '(b) verify employment/business continuity, '
            '(c) obtain credit committee sign-off before proceeding.',
        'Rejected':
            'Application declined. Borrower remediation advice: '
            '(a) target CIBIL score improvement to 700+, '
            '(b) reduce existing loan obligations to lower DTI, '
            '(c) accumulate savings reserves, '
            '(d) reapply after minimum 6-month remediation period.',
    }
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(0, 4.5, _clean(cond_text.get(verdict, '')))
    pdf.ln(5)

    # ── DISCLAIMER ─────────────────────────────────────────────────────────────
    pdf.set_fill_color(245, 247, 252)
    pdf.set_draw_color(200, 204, 240)
    disc_y = pdf.get_y()
    pdf.rect(20, disc_y, 170, 1, 'FD')
    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 3.5,
        'DISCLAIMER: This credit committee memorandum is produced by CreditIQ, an AI-assisted credit risk '
        'underwriting prototype developed as an MBA Finance Live Project. All borrower data used is '
        'synthetic and does not represent any real customer. This document does not constitute a binding '
        'lending decision or regulatory compliance opinion. Intended for educational and demonstration purposes only.')

    return bytes(pdf.output())


def generate_credit_memo_pdf_full(
    details, cs_scores, w_score, grade, verdict,
    ml_pd, ml_confidence, stops, flags,
    analyst_notes, pos_factors, neg_factors,
    consensus_agree, consensus_explanation, grade_label='',
    audit_rows=None,
) -> bytes:
    """
    Full version with policy audit table rows injected.
    audit_rows: list of (rule, passed, value, rationale)
    """
    today  = datetime.date.today()
    ref_id = f"CIQ-{today.strftime('%Y%m%d')}-{abs(hash(str(details))) % 9999:04d}"

    if verdict == 'Approved':
        rec_rgb = (16, 185, 129)
    elif verdict in ('Approved with Conditions', 'Manual Review'):
        rec_rgb = (217, 119, 6)
    else:
        rec_rgb = (220, 38, 38)

    pdf = _MemoPDF(ref_id)
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 13)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, 'CREDIT COMMITTEE MEMORANDUM', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5,
             f'Date: {today.strftime("%d %B %Y")}  |  AI Underwriting Engine  |  Ref: {ref_id}',
             align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    # Executive summary box
    box_y  = pdf.get_y()
    pdf.set_fill_color(245, 247, 252)
    pdf.set_draw_color(99, 102, 241)
    pdf.rect(20, box_y, 170, 24, 'FD')
    col_w     = 42
    ex_labels = ['Risk Score', 'Risk Grade', 'Default Prob.', 'Decision']
    ex_vals   = [f'{w_score:.0f}/100', grade, f'{ml_pd*100:.1f}%', verdict[:18]]
    pd_clr    = (16, 185, 129) if ml_pd < 0.15 else (217, 119, 6) if ml_pd < 0.30 else (220, 38, 38)
    ex_clrs   = [(79, 70, 229), (79, 70, 229), pd_clr, rec_rgb]
    pdf.set_y(box_y + 3); pdf.set_x(20)
    pdf.set_font('Helvetica', '', 7); pdf.set_text_color(100, 116, 139)
    for l in ex_labels: pdf.cell(col_w, 4, l, align='C')
    pdf.ln(4); pdf.set_x(20)
    for v, c in zip(ex_vals, ex_clrs):
        pdf.set_font('Helvetica', 'B', 11); pdf.set_text_color(*c)
        pdf.cell(col_w, 9, _clean(v), align='C')
    pdf.ln(13)

    # ── Section 1: Applicant ─────────────────────────────────────────────────
    pdf.section_header('1.  APPLICANT PROFILE')
    cat_lbl = {'Salaried': 'Salaried Professional', 'Self-Employed': 'Self-Employed',
               'Student': 'Student', 'Retired': 'Retired'}
    for label, val in [
        ('Borrower Category', cat_lbl.get(details.get('Category',''), details.get('Category',''))),
        ('Age',               f"{details.get('Age','-')} years"),
        ('Monthly Income',    f"Rs. {details.get('Monthly_Income',0):,.0f}"),
        ('Loan Amount',       f"Rs. {details.get('Loan_Amount',0):,.0f}"),
        ('Loan Purpose',      details.get('Loan_Purpose','-')),
        ('Tenure',            f"{details.get('Loan_Tenure',0)} months"),
        ('Co-Applicant',      'Yes' if details.get('Co_Applicant',0) else 'No'),
        ('CIBIL Score',       str(details.get('CIBIL_Score','-'))),
    ]:
        pdf.kv_row(label, val)
    pdf.ln(3)

    # ── Section 2: Risk Assessment ───────────────────────────────────────────
    pdf.section_header('2.  CREDIT RISK ASSESSMENT')
    for label, val in [
        ('Rule Score (5 Cs)',        f'{w_score:.1f} / 100'),
        ('Risk Grade',               f'{grade} — {grade_label}'),
        ('ML Default Probability',   f'{ml_pd*100:.1f}%'),
        ('ML Confidence',            f'{ml_confidence:.1f}%'),
        ('DTI Ratio',                f"{details.get('DTI_Ratio',0):.1f}%"),
        ('Missed EMIs',              str(details.get('Missed_EMIs','-'))),
        ('LTI Ratio',                f"{details.get('Loan_To_Income_Ratio',0):.1f}x"),
        ('Model Consensus',          'AGREE' if consensus_agree else 'DISAGREE'),
    ]:
        pdf.kv_row(label, val)
    if not consensus_agree:
        pdf.ln(1); pdf.set_font('Helvetica','I',7.5); pdf.set_text_color(180,100,0)
        pdf.multi_cell(0, 4, _clean(f'Note: {consensus_explanation}'))
    pdf.ln(3)

    # ── Section 3: 5 Cs ─────────────────────────────────────────────────────
    pdf.section_header('3.  FIVE Cs SCORECARD')
    wm = {'Character':35,'Capacity':30,'Capital':15,'Collateral':15,'Conditions':5}
    for c_name, c_val in cs_scores.items():
        w_p = wm.get(c_name, 0)
        contrib = round(c_val * (w_p/100) * 10, 1)
        bar_tot = 90; filled = int(c_val/10*bar_tot)
        pdf.set_font('Helvetica','',8); pdf.set_text_color(30,41,59)
        pdf.cell(28,5,c_name); pdf.set_text_color(100,116,139); pdf.cell(10,5,f'{w_p}%',align='C')
        bx = pdf.get_x(); by = pdf.get_y()+1.5
        pdf.set_fill_color(220,222,240); pdf.rect(bx,by,bar_tot,3,'F')
        fc = (16,185,129) if c_val>=7 else (217,119,6) if c_val>=5 else (220,38,38)
        pdf.set_fill_color(*fc)
        if filled>0: pdf.rect(bx,by,filled,3,'F')
        pdf.set_x(bx+bar_tot+2); pdf.set_font('Helvetica','B',8); pdf.set_text_color(79,70,229)
        pdf.cell(14,5,f'{c_val:.1f}/10')
        pdf.set_font('Helvetica','',7); pdf.set_text_color(100,116,139)
        pdf.cell(0,5,f'(+{contrib} pts)', new_x='LMARGIN', new_y='NEXT')
    pdf.ln(3)

    # ── Section 4: Policy Audit Table ───────────────────────────────────────
    pdf.section_header('4.  UNDERWRITING POLICY COMPLIANCE')
    pdf.set_fill_color(241,245,249); pdf.set_font('Helvetica','B',8); pdf.set_text_color(30,41,59)
    pdf.cell(80,6,'Policy Rule',fill=True,border=1)
    pdf.cell(40,6,'Actual Value',fill=True,border=1,align='C')
    pdf.cell(50,6,'Status',fill=True,border=1,align='C',new_x='LMARGIN',new_y='NEXT')

    if audit_rows:
        for rule, passed, value, rationale in audit_rows:
            pdf.set_font('Helvetica','',8); pdf.set_text_color(30,41,59)
            pdf.cell(80,5,_clean(rule),border=1)
            pdf.cell(40,5,_clean(value),border=1,align='C')
            if passed:
                pdf.set_text_color(16,150,80)
                pdf.cell(50,5,'PASS',border=1,align='C',new_x='LMARGIN',new_y='NEXT')
            else:
                pdf.set_text_color(200,38,38)
                pdf.cell(50,5,'FAIL',border=1,align='C',new_x='LMARGIN',new_y='NEXT')
    pdf.ln(3)

    # ── Section 5: Strengths & Concerns ─────────────────────────────────────
    pdf.section_header('5.  KEY STRENGTHS & RISK CONCERNS')
    xl,xr,cw2 = 20,107,84
    pdf.set_xy(xl,pdf.get_y())
    pdf.set_font('Helvetica','B',8); pdf.set_text_color(16,150,100)
    pdf.cell(cw2,5,'KEY STRENGTHS')
    pdf.set_xy(xr,pdf.get_y())
    pdf.set_text_color(200,50,50); pdf.cell(cw2,5,'RISK CONCERNS'); pdf.ln(6)
    n = max(len(pos_factors[:5]), len(neg_factors[:5]))
    for i in range(n):
        ry = pdf.get_y(); ny = ry
        if i < len(pos_factors):
            pdf.set_xy(xl,ry); pdf.set_font('Helvetica','',7); pdf.set_text_color(30,41,59)
            pdf.multi_cell(cw2-2,4,_clean(f'+ {pos_factors[i]}'))
            ny = max(ny, pdf.get_y())
        if i < len(neg_factors):
            pdf.set_xy(xr,ry); pdf.set_font('Helvetica','',7); pdf.set_text_color(30,41,59)
            pdf.multi_cell(cw2-2,4,_clean(f'- {neg_factors[i]}'))
            ny = max(ny, pdf.get_y())
        pdf.set_y(ny+1)
    pdf.ln(3)

    # ── Section 6: Analyst Notes ─────────────────────────────────────────────
    pdf.section_header('6.  CREDIT ANALYST COMMENTARY')
    pdf.set_font('Helvetica','I',8.5); pdf.set_text_color(30,41,59)
    pdf.multi_cell(0,4.5,_clean(analyst_notes))
    pdf.ln(3)

    # ── Section 7: Recommendation ────────────────────────────────────────────
    pdf.section_header('7.  FINAL RECOMMENDATION')
    pdf.set_fill_color(*rec_rgb); pdf.set_text_color(255,255,255)
    pdf.set_font('Helvetica','B',13)
    pdf.cell(0,11,f'  {verdict.upper()}',fill=True,new_x='LMARGIN',new_y='NEXT')
    pdf.ln(3)

    conditions = {
        'Approved': 'Standard approval at best applicable rate. Proceed with documentation and disbursement.',
        'Approved with Conditions': 'Approval subject to: (a) enhanced rate margin, (b) co-applicant if DTI > 45%, (c) reduced quantum or staged disbursement, (d) annual income review.',
        'Manual Review': 'Refer to credit committee. Collect supplementary income proof and verify employment continuity before proceeding.',
        'Rejected': 'Decline advised. Borrower remediation: improve CIBIL to 700+, reduce DTI, build savings, reapply after 6 months.',
    }
    pdf.set_font('Helvetica','',8.5); pdf.set_text_color(30,41,59)
    pdf.multi_cell(0,4.5,_clean(conditions.get(verdict,'')))
    pdf.ln(4)

    # Disclaimer
    pdf.set_font('Helvetica','I',7); pdf.set_text_color(120,130,150)
    pdf.multi_cell(0,3.5,
        'DISCLAIMER: CreditIQ is an AI-assisted underwriting prototype (MBA Finance Live Project). '
        'All data is synthetic. Not a binding decision. For educational purposes only.')

    return bytes(pdf.output())
