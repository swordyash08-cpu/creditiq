"""
CreditIQ — Bulk Upload Engine
Handles CSV/XLSX file uploads for batch loan application processing.
Validates, standardises, scores, and prepares records for database insertion.
"""

import io
import datetime
import pandas as pd
import numpy as np

from model import (
    calculate_5cs_scores, calculate_weighted_score,
    check_hard_stops, check_red_flags, get_rule_based_decision,
    get_risk_grade, predict_single_probability,
    get_model_consensus,
)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN NAME MAPPING — normalise common header variations to internal names
# ══════════════════════════════════════════════════════════════════════════════

_COLUMN_MAP = {
    # Income variants
    'income':           'Monthly_Income',
    'monthly_income':   'Monthly_Income',
    'monthlyincome':    'Monthly_Income',
    'monthly income':   'Monthly_Income',
    'net_income':       'Monthly_Income',
    # CIBIL / Credit Score variants
    'credit_score':     'CIBIL_Score',
    'cibil_score':      'CIBIL_Score',
    'creditscore':      'CIBIL_Score',
    'cibil':            'CIBIL_Score',
    'cibilscore':       'CIBIL_Score',
    'credit score':     'CIBIL_Score',
    # Category / Employment Type variants
    'employment_type':  'Category',
    'category':         'Category',
    'borrower_type':    'Category',
    'employmenttype':   'Category',
    'borrowertype':     'Category',
    'employment type':  'Category',
    # Employment Length / Stability variants
    'employment_stability': 'Employment_Length',
    'employment_length':    'Employment_Length',
    'employmentstability':  'Employment_Length',
    'employmentlength':     'Employment_Length',
    'emp_length':           'Employment_Length',
    'work_experience':      'Employment_Length',
    # DTI variants
    'dti_ratio':        'DTI_Ratio',
    'debt_to_income':   'DTI_Ratio',
    'dtiratio':         'DTI_Ratio',
    'dti':              'DTI_Ratio',
    'debt to income':   'DTI_Ratio',
    # Missed EMIs / Defaults variants
    'previous_defaults': 'Missed_EMIs',
    'missed_emis':       'Missed_EMIs',
    'previousdefaults':  'Missed_EMIs',
    'missedemis':        'Missed_EMIs',
    'defaults':          'Missed_EMIs',
    'missed_payments':   'Missed_EMIs',
    # Standard fields (lower-case normalisations)
    'applicant_id':          'Applicant_ID',
    'applicantid':           'Applicant_ID',
    'application_id':        'Applicant_ID',
    'age':                   'Age',
    'savings':               'Savings',
    'existing_emis':         'Existing_EMIs',
    'existingemis':          'Existing_EMIs',
    'existing_emi':          'Existing_EMIs',
    'loan_amount':           'Loan_Amount',
    'loanamount':            'Loan_Amount',
    'loan amount':           'Loan_Amount',
    'loan_tenure':           'Loan_Tenure',
    'loantenure':            'Loan_Tenure',
    'loan tenure':           'Loan_Tenure',
    'tenure':                'Loan_Tenure',
    'education_level':       'Education_Level',
    'educationlevel':        'Education_Level',
    'education':             'Education_Level',
    'residence_type':        'Residence_Type',
    'residencetype':         'Residence_Type',
    'residence':             'Residence_Type',
    # Optional but useful if present
    'investments':           'Investments',
    'asset_value':           'Asset_Value',
    'assetvalue':            'Asset_Value',
    'loan_purpose':          'Loan_Purpose',
    'loanpurpose':           'Loan_Purpose',
    'co_applicant':          'Co_Applicant',
    'coapplicant':           'Co_Applicant',
    'credit_history_length': 'Credit_History_Length',
    'credithistorylength':   'Credit_History_Length',
    'credit_history':        'Credit_History_Length',
    'net_worth':             'Net_Worth',
    'networth':              'Net_Worth',
}

REQUIRED_FIELDS = [
    'Applicant_ID', 'Age', 'Monthly_Income', 'Category',
    'Employment_Length', 'CIBIL_Score', 'DTI_Ratio', 'Savings',
    'Existing_EMIs', 'Loan_Amount', 'Loan_Tenure', 'Missed_EMIs',
    'Education_Level', 'Residence_Type',
]

TEMPLATE_ROWS = [
    {
        'Applicant_ID': 'APP-00001', 'Age': 35, 'Monthly_Income': 85000,
        'Category': 'Salaried', 'Employment_Length': 6.5, 'CIBIL_Score': 760,
        'DTI_Ratio': 28.5, 'Savings': 320000, 'Existing_EMIs': 18000,
        'Loan_Amount': 1500000, 'Loan_Tenure': 60, 'Missed_EMIs': 0,
        'Education_Level': 'Graduate', 'Residence_Type': 'Owned',
    },
    {
        'Applicant_ID': 'APP-00002', 'Age': 42, 'Monthly_Income': 55000,
        'Category': 'Self-Employed', 'Employment_Length': 10.0, 'CIBIL_Score': 690,
        'DTI_Ratio': 44.2, 'Savings': 150000, 'Existing_EMIs': 24000,
        'Loan_Amount': 800000, 'Loan_Tenure': 36, 'Missed_EMIs': 1,
        'Education_Level': 'Post-Graduate', 'Residence_Type': 'Rented',
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — COLUMN STANDARDISATION
# ══════════════════════════════════════════════════════════════════════════════

def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Maps uploaded column names to internal standard names (case-insensitive)."""
    rename = {}
    for col in df.columns:
        lookup = col.strip().lower().replace(' ', '_')
        # Try progressively relaxed matches
        if lookup in _COLUMN_MAP:
            rename[col] = _COLUMN_MAP[lookup]
        elif lookup.replace('_', '') in _COLUMN_MAP:
            rename[col] = _COLUMN_MAP[lookup.replace('_', '')]
        # If already matches an internal name exactly, keep it
        elif col in REQUIRED_FIELDS:
            rename[col] = col
    return df.rename(columns=rename)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — VALIDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_VALIDATION_RULES = {
    'Age':           {'min': 18,  'max': 75,   'dtype': 'numeric'},
    'CIBIL_Score':   {'min': 300, 'max': 900,  'dtype': 'numeric'},
    'Monthly_Income':{'min': 0,   'max': None, 'dtype': 'numeric'},
    'DTI_Ratio':     {'min': 0,   'max': 100,  'dtype': 'numeric'},
    'Loan_Amount':   {'min': 1,   'max': None, 'dtype': 'numeric'},
    'Loan_Tenure':   {'min': 6,   'max': 360,  'dtype': 'numeric'},
    'Missed_EMIs':   {'min': 0,   'max': None, 'dtype': 'numeric'},
    'Existing_EMIs': {'min': 0,   'max': None, 'dtype': 'numeric'},
    'Savings':       {'min': 0,   'max': None, 'dtype': 'numeric'},
}


def validate_upload(df: pd.DataFrame) -> dict:
    """
    Validates an uploaded DataFrame for batch processing.

    Returns
    -------
    dict
        valid            : bool — True if no blocking errors found
        total_records    : int
        missing_fields   : list of field names absent from upload
        duplicate_ids    : list of duplicated Applicant_ID values
        invalid_rows     : list of {row, field, value, reason} dicts
        quality_score    : float 0–100
        warnings         : list of human-readable warning strings
        cleaned_df       : pd.DataFrame with standardised column names
    """
    cleaned = _standardise_columns(df.copy())
    total = len(cleaned)
    warnings = []
    invalid_rows = []

    # ── Missing fields ────────────────────────────────────────────────────
    present = set(cleaned.columns)
    missing_fields = [f for f in REQUIRED_FIELDS if f not in present]
    if missing_fields:
        warnings.append(
            f"⚠️ Missing required columns: {', '.join(missing_fields)}. "
            f"These fields will use default values during processing."
        )

    # ── Duplicate Applicant_IDs ───────────────────────────────────────────
    duplicate_ids = []
    if 'Applicant_ID' in cleaned.columns:
        dupes = cleaned[cleaned['Applicant_ID'].duplicated(keep=False)]
        duplicate_ids = dupes['Applicant_ID'].unique().tolist()
        if duplicate_ids:
            warnings.append(
                f"⚠️ {len(duplicate_ids)} duplicate Applicant_ID(s) detected: "
                f"{', '.join(str(d) for d in duplicate_ids[:5])}"
                f"{'…' if len(duplicate_ids) > 5 else ''}"
            )

    # ── Row-level validation ──────────────────────────────────────────────
    valid_cells = 0
    total_cells = 0

    for idx, row in cleaned.iterrows():
        row_num = idx + 2  # Excel-style: header=1, data starts at 2
        for field, rules in _VALIDATION_RULES.items():
            if field not in cleaned.columns:
                continue
            total_cells += 1
            value = row.get(field)

            # Check for null / NaN
            if pd.isna(value):
                invalid_rows.append({
                    'row': row_num, 'field': field,
                    'value': None, 'reason': 'Missing value',
                })
                continue

            # Check numeric type
            try:
                num = float(value)
            except (ValueError, TypeError):
                invalid_rows.append({
                    'row': row_num, 'field': field,
                    'value': str(value), 'reason': 'Non-numeric value',
                })
                continue

            # Range checks
            if rules['min'] is not None and num < rules['min']:
                invalid_rows.append({
                    'row': row_num, 'field': field,
                    'value': num,
                    'reason': f"Below minimum ({rules['min']})",
                })
                continue

            if rules['max'] is not None and num > rules['max']:
                invalid_rows.append({
                    'row': row_num, 'field': field,
                    'value': num,
                    'reason': f"Exceeds maximum ({rules['max']})",
                })
                continue

            valid_cells += 1

    # ── Data quality score ────────────────────────────────────────────────
    quality_score = round((valid_cells / total_cells) * 100, 1) if total_cells > 0 else 0.0

    if quality_score < 70:
        warnings.append(
            f"⚠️ Data quality score is low ({quality_score}%). "
            f"Review flagged rows before processing."
        )

    # ── Summary warnings ──────────────────────────────────────────────────
    if total == 0:
        warnings.append("⚠️ Uploaded file contains no data rows.")

    if invalid_rows:
        warnings.append(
            f"⚠️ {len(invalid_rows)} cell-level validation issue(s) found across "
            f"{len(set(r['row'] for r in invalid_rows))} row(s)."
        )

    valid = (
        len(missing_fields) == 0
        and len(duplicate_ids) == 0
        and len(invalid_rows) == 0
        and total > 0
    )

    return {
        'valid':          valid,
        'total_records':  total,
        'missing_fields': missing_fields,
        'duplicate_ids':  duplicate_ids,
        'invalid_rows':   invalid_rows,
        'quality_score':  quality_score,
        'warnings':       warnings,
        'cleaned_df':     cleaned,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BATCH PROCESSING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

_CATEGORY_DEFAULTS = {'Salaried', 'Self-Employed', 'Student', 'Retired'}
_PURPOSE_DEFAULTS  = {'Home Purchase', 'Education', 'Business Expansion', 'Medical', 'Personal'}


def _build_borrower_dict(row: pd.Series) -> dict:
    """Converts a cleaned DataFrame row into a model-compatible borrower dict."""
    income = float(row.get('Monthly_Income', 0))
    loan   = float(row.get('Loan_Amount', 0))
    cat    = str(row.get('Category', 'Salaried'))
    if cat not in _CATEGORY_DEFAULTS:
        cat = 'Salaried'

    purpose = str(row.get('Loan_Purpose', 'Personal'))
    if purpose not in _PURPOSE_DEFAULTS:
        purpose = 'Personal'

    d = {
        'Category':              cat,
        'Age':                   int(row.get('Age', 30)),
        'Monthly_Income':        income,
        'Savings':               float(row.get('Savings', 0)),
        'Investments':           float(row.get('Investments', 0)),
        'CIBIL_Score':           int(row.get('CIBIL_Score', 650)),
        'Missed_EMIs':           int(row.get('Missed_EMIs', 0)),
        'Credit_History_Length': int(row.get('Credit_History_Length', 3)),
        'Existing_EMIs':         float(row.get('Existing_EMIs', 0)),
        'DTI_Ratio':             float(row.get('DTI_Ratio', 30)),
        'Employment_Length':     float(row.get('Employment_Length', 2)),
        'Asset_Value':           float(row.get('Asset_Value', 0)),
        'Loan_Amount':           loan,
        'Loan_Tenure':           int(row.get('Loan_Tenure', 36)),
        'Loan_Purpose':          purpose,
        'Co_Applicant':          int(row.get('Co_Applicant', 0)),
    }

    # Derived features
    annual_income = income * 12
    d['Loan_To_Income_Ratio'] = round(loan / annual_income, 2) if annual_income > 0 else 5.0
    d['Net_Worth'] = round(d['Savings'] + d['Investments'] + d['Asset_Value'] - loan, 2)
    return d


def _decision_explanation(verdict, grade, w_score, ml_pd, flags):
    """Generates a concise decision explanation string."""
    parts = [f"Risk Grade {grade} | Score {w_score:.1f}/100 | ML PD {ml_pd*100:.1f}%"]
    if verdict == 'Rejected':
        parts.append("Application does not meet credit policy thresholds.")
    elif verdict == 'Manual Review':
        parts.append("Borderline profile — referred for manual underwriting review.")
    elif verdict == 'Approved with Conditions':
        parts.append("Conditionally approved — enhanced pricing or collateral may apply.")
    else:
        parts.append("Profile meets all credit policy requirements.")
    if flags:
        parts.append(f"Risk flags: {'; '.join(flags[:3])}")
    return ' | '.join(parts)


def process_upload(df: pd.DataFrame, model_data: dict, batch_id: str) -> dict:
    """
    Processes validated applications through the full risk engine.

    Parameters
    ----------
    df          : pd.DataFrame — cleaned & validated (from validate_upload)
    model_data  : dict — trained model artefact from model.train_ml_model()
    batch_id    : str — unique identifier for this upload batch

    Returns
    -------
    dict
        processed     : int — total rows processed
        accepted      : int — Approved + Approved with Conditions
        rejected      : int
        conditional   : int — Approved with Conditions only
        manual_review : int
        avg_pd        : float — mean PD across batch
        avg_score     : float — mean weighted score
        records       : list of dicts ready for database.insert_applications_batch()
    """
    records = []
    pd_values = []
    scores = []
    counts = {'accepted': 0, 'rejected': 0, 'conditional': 0, 'manual_review': 0}

    for idx, row in df.iterrows():
        borrower = _build_borrower_dict(row)
        app_id = str(row.get('Applicant_ID', f'BULK-{batch_id}-{idx+1:05d}'))

        # ── 5Cs Scorecard ─────────────────────────────────────────────────
        cs = calculate_5cs_scores(borrower)
        w_score = calculate_weighted_score(cs)
        grade = get_risk_grade(w_score)

        # ── Policy checks ─────────────────────────────────────────────────
        hard_stops = check_hard_stops(borrower)
        flags, flag_severity = check_red_flags(borrower)
        rule_verdict, risk_cat, detail_msg = get_rule_based_decision(
            w_score, hard_stops, flag_severity,
        )

        # ── ML probability of default ─────────────────────────────────────
        ml_pd = 0.0
        try:
            ml_pd = predict_single_probability(model_data, borrower)
        except Exception:
            ml_pd = 0.15  # Fallback: assume moderate risk if model fails

        # ── Consensus decision ─────────────────────────────────────────────
        _, consensus_expl, final_verdict, decision_source = get_model_consensus(
            rule_verdict, ml_pd, w_score,
        )
        explanation = _decision_explanation(final_verdict, grade, w_score, ml_pd, flags)

        # ── Tally counters ─────────────────────────────────────────────────
        if final_verdict == 'Approved':
            counts['accepted'] += 1
        elif final_verdict == 'Approved with Conditions':
            counts['accepted'] += 1
            counts['conditional'] += 1
        elif final_verdict == 'Manual Review':
            counts['manual_review'] += 1
        else:
            counts['rejected'] += 1

        pd_values.append(ml_pd)
        scores.append(w_score)

        # ── Build database-ready record ────────────────────────────────────
        records.append({
            'applicant_id':          app_id,
            'category':              borrower['Category'],
            'age':                   borrower['Age'],
            'monthly_income':        borrower['Monthly_Income'],
            'savings':               borrower['Savings'],
            'investments':           borrower['Investments'],
            'cibil_score':           borrower['CIBIL_Score'],
            'missed_emis':           borrower['Missed_EMIs'],
            'credit_history_length': borrower['Credit_History_Length'],
            'existing_emis':         borrower['Existing_EMIs'],
            'dti_ratio':             borrower['DTI_Ratio'],
            'employment_length':     borrower['Employment_Length'],
            'asset_value':           borrower['Asset_Value'],
            'loan_amount':           borrower['Loan_Amount'],
            'loan_tenure':           borrower['Loan_Tenure'],
            'loan_purpose':          borrower['Loan_Purpose'],
            'co_applicant':          borrower['Co_Applicant'],
            'loan_to_income_ratio':  borrower['Loan_To_Income_Ratio'],
            'net_worth':             borrower['Net_Worth'],
            'risk_score':            round(w_score, 2),
            'pd_value':              round(ml_pd, 4),
            'risk_grade':            grade,
            'decision':              final_verdict,
            'decision_explanation':  explanation,
            'source':                'upload',
            'upload_batch_id':       batch_id,
        })

    processed = len(records)
    return {
        'processed':     processed,
        'accepted':      counts['accepted'],
        'rejected':      counts['rejected'],
        'conditional':   counts['conditional'],
        'manual_review': counts['manual_review'],
        'avg_pd':        round(float(np.mean(pd_values)), 4) if pd_values else 0.0,
        'avg_score':     round(float(np.mean(scores)), 1) if scores else 0.0,
        'records':       records,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TEMPLATE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_template_csv() -> bytes:
    """
    Returns a downloadable CSV template as bytes.
    Includes column headers with 2 example rows demonstrating expected formats.
    """
    template_df = pd.DataFrame(TEMPLATE_ROWS)
    # Ensure column order matches REQUIRED_FIELDS
    ordered_cols = [c for c in REQUIRED_FIELDS if c in template_df.columns]
    remaining = [c for c in template_df.columns if c not in ordered_cols]
    template_df = template_df[ordered_cols + remaining]

    buf = io.BytesIO()
    template_df.to_csv(buf, index=False, encoding='utf-8')
    return buf.getvalue()
