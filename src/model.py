import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RULE-BASED CREDIT SCORECARD (5 Cs of Credit)
# Weight structure: Character 35% | Capacity 30% | Capital 15% |
#                  Collateral 15% | Conditions 5%
# Finance justification:
#   • Character (35%) — CIBIL is the strongest predictor of default per RBI data
#   • Capacity  (30%) — DTI + income stability drive actual repayment ability
#   • Capital   (15%) — Savings buffer directly reduces Loss-Given-Default (LGD)
#   • Collateral(15%) — Asset coverage drives recovery rate in event of default
#   • Conditions (5%) — Loan purpose/tenure secondary to repayment ability
# ══════════════════════════════════════════════════════════════════════════════

def _score_cibil(cibil):
    """CIBIL/Credit Score → component score 1–10."""
    if cibil >= 800: return 10
    elif cibil >= 750: return 9
    elif cibil >= 700: return 7
    elif cibil >= 650: return 5
    elif cibil >= 600: return 3
    else: return 1

def _score_missed_emis(missed):
    """Payment delinquency → component score 1–10."""
    if missed == 0: return 10
    elif missed == 1: return 7
    elif missed == 2: return 4
    elif missed == 3: return 2
    else: return 1

def _score_credit_history(years):
    """Credit history length in years → component score 1–10."""
    if years >= 10: return 10
    elif years >= 7: return 8
    elif years >= 5: return 7
    elif years >= 3: return 5
    elif years >= 1: return 3
    else: return 1

def _score_income(income):
    """Monthly net income (₹) → component score 1–10."""
    if income >= 150000: return 10
    elif income >= 100000: return 9
    elif income >= 60000: return 7
    elif income >= 40000: return 5
    elif income >= 20000: return 3
    else: return 1

def _score_dti(dti):
    """Debt-to-Income ratio (%) → component score 1–10."""
    if dti <= 20: return 10
    elif dti <= 30: return 9
    elif dti <= 40: return 7
    elif dti <= 50: return 5
    elif dti <= 60: return 3
    else: return 1

def _score_employment(emp_length, category):
    """Employment/business vintage (years) → component score 1–10."""
    if category in ('Student', 'Retired'):
        return 8  # Exempt from job-tenure penalty
    if emp_length >= 8: return 10
    elif emp_length >= 5: return 8
    elif emp_length >= 3: return 6
    elif emp_length >= 1: return 4
    else: return 2

def _score_savings(savings, income):
    """Savings-to-annual-income ratio → component score 1–10."""
    annual = income * 12
    if annual == 0:
        return 8 if savings > 100000 else 2
    ratio = savings / annual
    if ratio >= 0.50: return 10
    elif ratio >= 0.30: return 8
    elif ratio >= 0.20: return 7
    elif ratio >= 0.10: return 5
    elif ratio >= 0.05: return 3
    else: return 1

def _score_investments(investments):
    """Investment portfolio (₹) → component score 1–10."""
    if investments >= 1000000: return 10
    elif investments >= 500000: return 8
    elif investments >= 200000: return 6
    elif investments >= 50000: return 4
    elif investments > 0: return 2
    else: return 1

def _score_collateral(loan_amount, asset_value):
    """LTV ratio → component score 1–10. Unsecured = 1."""
    if asset_value == 0:
        return 1  # Fully unsecured
    if loan_amount == 0:
        return 10
    ltv = (loan_amount / asset_value) * 100
    if ltv <= 40: return 10
    elif ltv <= 60: return 8
    elif ltv <= 75: return 6
    elif ltv <= 90: return 4
    elif ltv <= 100: return 2
    else: return 1

def _score_conditions(loan_purpose, tenure_months, loan_to_income):
    """Loan purpose + tenure + loan-to-income → component score 1–10."""
    purpose_map = {
        'Home Purchase': 9,
        'Education': 9,
        'Business Expansion': 7,
        'Medical': 6,
        'Personal': 5,
    }
    purpose_score = purpose_map.get(loan_purpose, 5)

    if tenure_months <= 36: tenure_score = 10
    elif tenure_months <= 60: tenure_score = 8
    elif tenure_months <= 120: tenure_score = 6
    else: tenure_score = 4

    if loan_to_income <= 2: lti_score = 10
    elif loan_to_income <= 4: lti_score = 7
    elif loan_to_income <= 6: lti_score = 5
    else: lti_score = 2

    return round(purpose_score * 0.40 + tenure_score * 0.30 + lti_score * 0.30, 2)


def calculate_5cs_scores(details):
    """
    Returns sub-scores for the 5 Cs on a 1–10 scale.
    Required keys: Category, Monthly_Income, Savings, Investments,
                   CIBIL_Score, Missed_EMIs, Credit_History_Length,
                   DTI_Ratio, Employment_Length, Asset_Value,
                   Loan_Amount, Loan_Tenure, Loan_Purpose,
                   Loan_To_Income_Ratio (default 5 if missing)
    """
    lti = details.get('Loan_To_Income_Ratio', 5.0)

    # 1. Character (35%)
    character = round(
        _score_cibil(details['CIBIL_Score'])         * 0.50 +
        _score_missed_emis(details['Missed_EMIs'])   * 0.30 +
        _score_credit_history(details['Credit_History_Length']) * 0.20,
        2
    )

    # 2. Capacity (30%)
    capacity = round(
        _score_dti(details['DTI_Ratio'])             * 0.45 +
        _score_income(details['Monthly_Income'])     * 0.35 +
        _score_employment(details['Employment_Length'], details['Category']) * 0.20,
        2
    )

    # 3. Capital (15%)
    capital = round(
        _score_savings(details['Savings'], details['Monthly_Income']) * 0.60 +
        _score_investments(details['Investments'])                     * 0.40,
        2
    )

    # 4. Collateral (15%)
    collateral = round(
        _score_collateral(details['Loan_Amount'], details['Asset_Value']),
        2
    )

    # 5. Conditions (5%)
    conditions = _score_conditions(
        details['Loan_Purpose'], details['Loan_Tenure'], lti
    )

    return {
        'Character':  character,
        'Capacity':   capacity,
        'Capital':    capital,
        'Collateral': collateral,
        'Conditions': conditions,
    }


def calculate_weighted_score(cs_scores):
    """
    Weighted composite score → 0–100 scale.
    Weights: Character 35% | Capacity 30% | Capital 15% | Collateral 15% | Conditions 5%
    """
    raw = (
        cs_scores['Character']  * 0.35 +
        cs_scores['Capacity']   * 0.30 +
        cs_scores['Capital']    * 0.15 +
        cs_scores['Collateral'] * 0.15 +
        cs_scores['Conditions'] * 0.05
    )
    return round(raw * 10, 1)  # Scale 1–10 → 10–100


def get_risk_grade(score):
    """
    Converts numeric score to credit risk grade (A+ → D).
    Aligned with standard bank internal rating scales.
    """
    if score >= 88: return 'A+'
    elif score >= 80: return 'A'
    elif score >= 72: return 'B+'
    elif score >= 62: return 'B'
    elif score >= 50: return 'C'
    else: return 'D'


def get_risk_grade_label(grade):
    labels = {
        'A+': 'Prime — Lowest Risk',
        'A':  'Near-Prime — Low Risk',
        'B+': 'Standard — Moderate Risk',
        'B':  'Sub-Standard — Elevated Risk',
        'C':  'Speculative — High Risk',
        'D':  'Distressed — Very High Risk',
    }
    return labels.get(grade, '')


# ── Hard Stops (Auto-Reject triggers) ─────────────────────────────────────────

HARD_STOPS = [
    ('CIBIL_Score',    lambda d: d['CIBIL_Score'] < 600,
     'CIBIL score {:.0f} is below minimum threshold of 600. Automatic rejection.'),
    ('DTI_Ratio',      lambda d: d['DTI_Ratio'] > 65.0,
     'Debt-to-Income ratio {:.1f}% exceeds maximum policy limit of 65%. Automatic rejection.'),
    ('Missed_EMIs',    lambda d: d['Missed_EMIs'] >= 4,
     '{:.0f} missed payments detected. Exceeds tolerance of 3. Automatic rejection.'),
    ('Income_Zero',    lambda d: d['Monthly_Income'] == 0 and d.get('Co_Applicant', 0) == 0,
     'Zero income with no co-applicant. Insufficient repayment capacity.'),
    ('LTI_Extreme',   lambda d: d.get('Loan_To_Income_Ratio', 0) > 10,
     'Loan-to-Income ratio {:.1f}x exceeds maximum policy limit of 10x. Loan amount too large.'),
]

def check_hard_stops(details):
    """Returns list of triggered hard-stop messages."""
    stops = []
    for key, condition, template in HARD_STOPS:
        if condition(details):
            val = details.get(key, details.get('DTI_Ratio', 0))
            try:
                stops.append(template.format(val))
            except Exception:
                stops.append(template)
    return stops


# ── Red Flags (Risk escalation signals) ───────────────────────────────────────

RED_FLAGS = [
    (lambda d: d['CIBIL_Score'] < 650,                                     2, 'CIBIL Score < 650 — poor repayment history'),
    (lambda d: d['DTI_Ratio'] > 50.0,                                      2, 'Post-loan DTI > 50% — strained repayment capacity'),
    (lambda d: d['Monthly_Income'] < 20000 and d['Category'] == 'Salaried',1, 'Low income < ₹20,000 for salaried category'),
    (lambda d: d['Savings'] < 20000,                                        1, 'Savings < ₹20,000 — very thin liquidity buffer'),
    (lambda d: d['Credit_History_Length'] < 2,                             1, 'Credit history < 2 years — thin credit file'),
    (lambda d: d.get('Loan_To_Income_Ratio', 0) > 6,                       1, 'Loan-to-Income ratio > 6x — large debt relative to income'),
    (lambda d: d['Missed_EMIs'] >= 2,                                       2, '2+ missed payments in last 24 months'),
    (lambda d: d['Employment_Length'] < 1 and d['Category'] not in ('Student', 'Retired'), 1,
     'Employment tenure < 1 year — job instability risk'),
]

def check_red_flags(details):
    """Returns (list of flag messages, cumulative severity score)."""
    flags, score = [], 0
    for condition, weight, message in RED_FLAGS:
        if condition(details):
            flags.append(message)
            score += weight
    return flags, score


def get_rule_based_decision(final_score, hard_stops, flag_count):
    """
    Returns (verdict, risk_category, detail_message).
    Decision hierarchy: Hard Stops → Red Flags → Score thresholds
    """
    if hard_stops:
        return ('Rejected', 'High Risk',
                'Hard Stop Triggered — Policy threshold violated. Manual review required.')

    if flag_count >= 5:
        return ('Rejected', 'High Risk',
                f'High Risk — {flag_count} risk flags detected. Profile does not meet credit policy.')
    elif flag_count >= 3:
        return ('Manual Review', 'Medium-High Risk',
                f'Elevated Risk — {flag_count} flags. Refer to senior underwriter for manual review.')
    elif flag_count >= 2:
        return ('Approved with Conditions', 'Medium Risk',
                f'Conditional Approval — {flag_count} risk flags. Apply enhanced pricing or co-applicant.')

    if final_score >= 85:
        return ('Approved', 'Low Risk',
                'Prime Profile — Excellent creditworthiness. Standard approval at best rate.')
    elif final_score >= 72:
        return ('Approved', 'Low-Medium Risk',
                'Good Profile — Approved at standard rate and terms.')
    elif final_score >= 60:
        return ('Approved with Conditions', 'Medium Risk',
                'Fair Profile — Approved with enhanced margin or reduced loan amount.')
    elif final_score >= 50:
        return ('Manual Review', 'Medium-High Risk',
                'Marginal Profile — Refer to underwriting committee for manual assessment.')
    else:
        return ('Rejected', 'High Risk',
                'Weak Profile — Score below minimum credit policy threshold.')


# ── Explainable AI: Analyst Notes ─────────────────────────────────────────────

def generate_analyst_notes(details, cs_scores, verdict, ml_pd):
    """
    Auto-generates a professional credit analyst underwriting summary.
    Mirrors real underwriter commentary used in credit appraisal memos.
    """
    name_map = {'Salaried': 'salaried professional', 'Self-Employed': 'self-employed individual',
                'Student': 'student applicant', 'Retired': 'retired individual'}
    profile = name_map.get(details.get('Category', 'Salaried'), 'applicant')

    cibil = details['CIBIL_Score']
    dti   = details['DTI_Ratio']
    income = details['Monthly_Income']
    savings = details['Savings']
    missed = details['Missed_EMIs']
    emp = details['Employment_Length']

    parts = []

    # Opening
    parts.append(
        f"The applicant is a {profile} with a CIBIL score of {cibil}."
    )

    # Character
    if cibil >= 750:
        parts.append("Credit history reflects strong repayment discipline and a prime risk profile.")
    elif cibil >= 700:
        parts.append("Repayment track record is satisfactory with no material delinquencies.")
    elif cibil >= 650:
        parts.append("Credit history exhibits minor irregularities warranting cautious assessment.")
    else:
        parts.append("Credit history shows significant delinquency risk requiring additional scrutiny.")

    if missed == 0:
        parts.append("Zero missed payments in the review period — consistent payment behaviour confirmed.")
    elif missed <= 2:
        parts.append(f"{missed} delayed payment(s) noted; risk impact is moderate and manageable.")
    else:
        parts.append(f"{missed} missed EMIs represent a material risk concern and elevate default probability.")

    # Capacity
    if dti <= 30:
        parts.append(f"Debt servicing burden is low at {dti:.1f}% DTI, indicating strong repayment capacity.")
    elif dti <= 50:
        parts.append(f"DTI of {dti:.1f}% is within acceptable range, though income headroom is moderate.")
    else:
        parts.append(f"Post-loan DTI of {dti:.1f}% is elevated and may constrain repayment under stress.")

    if income >= 80000:
        parts.append("Income level is robust and provides adequate buffer for loan servicing.")
    elif income >= 40000:
        parts.append("Income is sufficient to meet proposed loan obligations under normal conditions.")
    else:
        parts.append("Income adequacy is a limiting factor; stress resilience may be weak.")

    # Capital
    if savings >= income * 6:
        parts.append("Strong liquidity reserve (>6 months income) provides meaningful stress buffer.")
    elif savings >= income * 3:
        parts.append("Savings position is adequate, providing a moderate financial cushion.")
    else:
        parts.append("Savings buffer is thin, limiting financial resilience under adverse scenarios.")

    # Employment
    if details.get('Category') not in ('Student', 'Retired'):
        if emp >= 5:
            parts.append(f"Employment/business vintage of {emp:.1f} years indicates stability.")
        elif emp >= 2:
            parts.append(f"Employment tenure of {emp:.1f} years is reasonable but warrants monitoring.")
        else:
            parts.append(f"Short employment duration of {emp:.1f} years introduces income continuity risk.")

    # ML observation
    if ml_pd < 0.10:
        parts.append(f"Machine learning model corroborates low-risk classification with a predicted default probability of {ml_pd*100:.1f}%.")
    elif ml_pd < 0.25:
        parts.append(f"ML engine assigns a moderate default probability of {ml_pd*100:.1f}%, consistent with rule-based assessment.")
    else:
        parts.append(f"ML model flags elevated default risk at {ml_pd*100:.1f}% PD — divergence from rule engine noted for committee review.")

    # Closing recommendation
    closing_map = {
        'Approved':                 'Recommendation: Proceed with standard loan sanction.',
        'Approved with Conditions': 'Recommendation: Sanction with enhanced pricing, co-applicant requirement, or reduced limit.',
        'Manual Review':            'Recommendation: Escalate to credit committee for manual underwriting decision.',
        'Rejected':                 'Recommendation: Decline application. Advise applicant on improvement steps.',
    }
    parts.append(closing_map.get(verdict, ''))

    return ' '.join(parts)


# ── Stress Testing ─────────────────────────────────────────────────────────────

def run_stress_test(model_data, details, base_score, base_pd):
    """
    Runs 3 RBI-aligned stress scenarios and returns impact on PD and score.
    Scenarios:
      1. Income drops 20% (job loss / pay-cut scenario)
      2. Interest rate rises 200 bps (rate shock scenario)
      3. EMI burden rises 25% (additional loan / family obligation scenario)
    Returns list of dicts: {scenario, new_score, new_pd, score_delta, pd_delta}
    """
    from src.utils import calculate_emi
    scenarios = []

    def _evaluate(modified):
        cs = calculate_5cs_scores(modified)
        sc = calculate_weighted_score(cs)
        pd = predict_single_probability(model_data, modified)
        return round(sc, 1), round(pd, 4)

    # Scenario 1: Income shock −20%
    s1 = details.copy()
    s1['Monthly_Income'] = round(details['Monthly_Income'] * 0.80, 0)
    if s1['Monthly_Income'] > 0:
        s1['DTI_Ratio'] = round((s1['Existing_EMIs'] / s1['Monthly_Income']) * 100, 1)
    sc1, pd1 = _evaluate(s1)
    scenarios.append({
        'scenario':    '📉 Income Shock (−20%)',
        'description': 'Simulates job loss or pay-cut scenario',
        'new_score':   sc1,
        'new_pd':      pd1,
        'score_delta': round(sc1 - base_score, 1),
        'pd_delta':    round((pd1 - base_pd) * 100, 2),
    })

    # Scenario 2: Interest rate +200 bps
    s2 = details.copy()
    base_rate = 11.0  # assumed benchmark rate
    new_rate   = base_rate + 2.0
    if details.get('Loan_Amount', 0) > 0 and details.get('Loan_Tenure', 0) > 0:
        new_emi = calculate_emi(details['Loan_Amount'], new_rate, details['Loan_Tenure'])
        old_emi = calculate_emi(details['Loan_Amount'], base_rate, details['Loan_Tenure'])
        emi_increase = new_emi - old_emi
        s2['Existing_EMIs'] = details.get('Existing_EMIs', 0) + emi_increase
        if s2['Monthly_Income'] > 0:
            s2['DTI_Ratio'] = round((s2['Existing_EMIs'] / s2['Monthly_Income']) * 100, 1)
    sc2, pd2 = _evaluate(s2)
    scenarios.append({
        'scenario':    '📈 Rate Shock (+200 bps)',
        'description': 'Simulates RBI rate hike / floating rate revision',
        'new_score':   sc2,
        'new_pd':      pd2,
        'score_delta': round(sc2 - base_score, 1),
        'pd_delta':    round((pd2 - base_pd) * 100, 2),
    })

    # Scenario 3: EMI burden +25%
    s3 = details.copy()
    s3['Existing_EMIs'] = round(details.get('Existing_EMIs', 0) * 1.25, 0)
    if s3['Monthly_Income'] > 0:
        s3['DTI_Ratio'] = round((s3['Existing_EMIs'] / s3['Monthly_Income']) * 100, 1)
    sc3, pd3 = _evaluate(s3)
    scenarios.append({
        'scenario':    '💳 EMI Stress (+25% Burden)',
        'description': 'Simulates additional credit obligations or family expenses',
        'new_score':   sc3,
        'new_pd':      pd3,
        'score_delta': round(sc3 - base_score, 1),
        'pd_delta':    round((pd3 - base_pd) * 100, 2),
    })

    return scenarios


# ── Improvement Suggestions ────────────────────────────────────────────────────

def get_improvement_suggestions(details, cs_scores, current_pd, model_data):
    """
    Generates specific, quantified improvement suggestions for the borrower.
    Returns list of suggestion strings.
    """
    suggestions = []
    income = details['Monthly_Income']
    dti = details['DTI_Ratio']
    cibil = details['CIBIL_Score']
    savings = details['Savings']
    emis = details['Existing_EMIs']
    loan_amt = details['Loan_Amount']

    # CIBIL improvement
    if cibil < 750:
        test = details.copy()
        test['CIBIL_Score'] = min(cibil + 50, 900)
        new_pd = predict_single_probability(model_data, test)
        delta = round((current_pd - new_pd) * 100, 1)
        if delta > 0:
            suggestions.append(
                f"🎯 Improving CIBIL score by 50 points (to {test['CIBIL_Score']}) could reduce default probability by ~{delta}%."
            )

    # DTI reduction via loan amount cut
    if dti > 35 and loan_amt > 100000:
        reduced_loan = round(loan_amt * 0.80 / 10000) * 10000
        test = details.copy()
        test['Loan_Amount'] = reduced_loan
        new_pd = predict_single_probability(model_data, test)
        delta = round((current_pd - new_pd) * 100, 1)
        if delta > 0:
            suggestions.append(
                f"💰 Reducing loan amount by ₹{loan_amt - reduced_loan:,.0f} (to ₹{reduced_loan:,.0f}) could lower default probability by ~{delta}%."
            )

    # Pre-close existing EMIs
    if emis > income * 0.15:
        test = details.copy()
        reduced_emis = round(emis * 0.70, 0)
        test['Existing_EMIs'] = reduced_emis
        if income > 0:
            test['DTI_Ratio'] = round((reduced_emis / income) * 100, 1)
        new_pd = predict_single_probability(model_data, test)
        delta = round((current_pd - new_pd) * 100, 1)
        if delta > 0:
            suggestions.append(
                f"🔄 Pre-closing existing loans to reduce EMIs by 30% could reduce default probability by ~{delta}%."
            )

    # Savings improvement
    if savings < income * 3:
        test = details.copy()
        test['Savings'] = income * 4
        new_pd = predict_single_probability(model_data, test)
        delta = round((current_pd - new_pd) * 100, 1)
        if delta > 0:
            suggestions.append(
                f"🏦 Building savings to ₹{test['Savings']:,.0f} (4× monthly income) could improve creditworthiness by ~{delta}% reduction in PD."
            )

    # Tenure extension to reduce EMI
    tenure = details['Loan_Tenure']
    if dti > 40 and tenure < 120:
        from src.utils import calculate_emi
        new_tenure = min(tenure + 24, 180)
        new_emi = calculate_emi(loan_amt, 11.0, new_tenure)
        old_emi = calculate_emi(loan_amt, 11.0, tenure)
        if income > 0:
            new_dti = round(((emis - old_emi + new_emi) / income) * 100, 1)
            if new_dti < dti - 5:
                suggestions.append(
                    f"📅 Extending loan tenure by 24 months could reduce monthly EMI by ₹{old_emi - new_emi:,.0f}, improving DTI to {new_dti:.1f}%."
                )

    if not suggestions:
        suggestions.append("✅ Profile is well-optimised. No significant improvement levers identified.")

    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MACHINE LEARNING MODEL (Random Forest with Cross-Validation)
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {'Student': 0, 'Salaried': 1, 'Self-Employed': 2, 'Retired': 3}
PURPOSE_MAP  = {
    'Education': 0, 'Personal': 1, 'Home Purchase': 2,
    'Medical': 3, 'Business Expansion': 4
}

FEATURE_COLS = [
    'Category', 'Age', 'Monthly_Income', 'Savings', 'Investments',
    'CIBIL_Score', 'Missed_EMIs', 'Credit_History_Length',
    'Existing_EMIs', 'DTI_Ratio', 'Employment_Length', 'Asset_Value',
    'Loan_Amount', 'Loan_Tenure', 'Loan_Purpose', 'Co_Applicant',
    'Loan_To_Income_Ratio', 'Net_Worth'
]


def _encode_df(df):
    enc = df.copy()
    enc['Category']    = enc['Category'].map(CATEGORY_MAP).fillna(-1).astype(int)
    enc['Loan_Purpose'] = enc['Loan_Purpose'].map(PURPOSE_MAP).fillna(-1).astype(int)
    # Keep only feature columns that exist
    available = [c for c in FEATURE_COLS if c in enc.columns]
    return enc[available], available


def train_ml_model(csv_path, model_save_path):
    """
    Trains a Random Forest classifier with 5-fold stratified cross-validation.
    Saves model, mappings, metrics and feature importances to pickle.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data not found: {csv_path}. Run generator.py first.")

    df = pd.read_csv(csv_path)
    X_enc, feat_cols = _encode_df(df)
    y = df['Default']

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.20, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Evaluation
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_enc, y, cv=cv, scoring='roc_auc', n_jobs=-1)

    feature_importances = dict(
        sorted(
            zip(feat_cols, rf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    )

    model_data = {
        'model':               rf,
        'features':            feat_cols,
        'metrics': {
            'accuracy':   accuracy,
            'auc':        auc,
            'report':     report,
            'cv_auc_mean': round(float(cv_scores.mean()), 4),
            'cv_auc_std':  round(float(cv_scores.std()), 4),
        },
        'feature_importances': feature_importances,
    }

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✅ Random Forest trained | Accuracy: {accuracy*100:.2f}% | AUC: {auc:.4f}")
    print(f"   5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return model_data


def predict_single_probability(model_data, details):
    """Predicts P(Default) for a single applicant dict."""
    rf      = model_data['model']
    feat    = model_data['features']

    row = details.copy()
    row['Category']    = CATEGORY_MAP.get(row.get('Category', 'Salaried'), 1)
    row['Loan_Purpose'] = PURPOSE_MAP.get(row.get('Loan_Purpose', 'Personal'), 1)

    # Fill missing engineered features with safe defaults
    if 'Loan_To_Income_Ratio' not in row:
        income = row.get('Monthly_Income', 1)
        row['Loan_To_Income_Ratio'] = row.get('Loan_Amount', 0) / (income * 12) if income > 0 else 5.0
    if 'Net_Worth' not in row:
        row['Net_Worth'] = (row.get('Savings', 0) + row.get('Investments', 0) +
                            row.get('Asset_Value', 0) - row.get('Loan_Amount', 0))

    values = [row.get(c, 0) for c in feat]
    df_row = pd.DataFrame([values], columns=feat)
    return round(float(rf.predict_proba(df_row)[0][1]), 4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GOVERNANCE, EXPLAINABILITY & ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def get_model_consensus(rule_verdict, ml_pd, w_score):
    """
    Determines whether the rule-based engine and Random Forest agree.
    Returns: (agree: bool, explanation: str, final_decision: str, decision_source: str)
    """
    rule_ok = rule_verdict in ('Approved', 'Approved with Conditions')
    ml_ok   = ml_pd < 0.25   # < 25% PD = ML considers acceptable

    if rule_ok and ml_ok:
        return (True,
                'Both the rule-based scorecard and Random Forest model independently '
                'classify this profile as an acceptable credit risk. Consensus approval.',
                rule_verdict,
                'Rule Engine + Random Forest Consensus')
    elif not rule_ok and not ml_ok:
        return (True,
                f'Both engines independently flag elevated credit risk. '
                f'Rule engine score {w_score:.0f}/100 is below threshold; '
                f'RF predicts {ml_pd*100:.1f}% default probability.',
                rule_verdict,
                'Rule Engine + Random Forest Consensus')
    elif rule_ok and not ml_ok:
        return (False,
                f'Rule engine approves (score {w_score:.0f}/100), but Random Forest '
                f'flags elevated risk ({ml_pd*100:.1f}% PD). ML detects subtle risk '
                f'patterns — high DTI, income volatility, or loan concentration — '
                f'not captured by threshold rules. Manual review recommended.',
                'Manual Review',
                'Model Disagreement — Manual Review Required')
    else:  # not rule_ok and ml_ok
        return (False,
                f'ML model predicts low default probability ({ml_pd*100:.1f}%), '
                f'but rule engine applies strict policy thresholds (score {w_score:.0f}/100). '
                f'Borderline scorecard case — may qualify under enhanced conditions.',
                'Approved with Conditions',
                'ML Signal — Conditional Approval')


def get_confidence_score(model_data, details):
    """Returns the RF model's confidence (max class probability) as a percentage."""
    rf   = model_data['model']
    feat = model_data['features']

    row = details.copy()
    row['Category']     = CATEGORY_MAP.get(row.get('Category', 'Salaried'), 1)
    row['Loan_Purpose'] = PURPOSE_MAP.get(row.get('Loan_Purpose', 'Personal'), 1)
    if 'Loan_To_Income_Ratio' not in row:
        inc = row.get('Monthly_Income', 1)
        row['Loan_To_Income_Ratio'] = row.get('Loan_Amount', 0) / (inc * 12) if inc > 0 else 5.0
    if 'Net_Worth' not in row:
        row['Net_Worth'] = (row.get('Savings', 0) + row.get('Investments', 0) +
                            row.get('Asset_Value', 0) - row.get('Loan_Amount', 0))
    values = [row.get(c, 0) for c in feat]
    df_row = pd.DataFrame([values], columns=feat)
    proba  = rf.predict_proba(df_row)[0]
    return round(float(max(proba)) * 100, 1)


def get_weighted_contributions(cs_scores):
    """
    Returns actual point contribution of each C to the final 100-point score.
    Character×35 + Capacity×30 + Capital×15 + Collateral×15 + Conditions×5 = 100
    """
    weights = {'Character': 0.35, 'Capacity': 0.30, 'Capital': 0.15,
               'Collateral': 0.15, 'Conditions': 0.05}
    return {c: round(cs_scores[c] * weights[c] * 10, 1) for c in cs_scores}


# ── Hard-Stop Policy Audit (structured table) ─────────────────────────────────

_POLICY_RULES = [
    ('CIBIL Score ≥ 650',        lambda d: d['CIBIL_Score'] >= 650,
     lambda d: str(d['CIBIL_Score']),
     'Minimum credit score threshold for retail lending.'),
    ('DTI Ratio ≤ 60%',          lambda d: d['DTI_Ratio'] <= 60.0,
     lambda d: f"{d['DTI_Ratio']:.1f}%",
     'Maximum debt servicing burden allowed by policy.'),
    ('Missed EMIs < 3',           lambda d: d['Missed_EMIs'] < 3,
     lambda d: str(d['Missed_EMIs']),
     'More than 2 missed payments indicates active delinquency risk.'),
    ('Income ≥ ₹15,000/month',   lambda d: d['Monthly_Income'] >= 15000,
     lambda d: f"₹{d['Monthly_Income']:,.0f}",
     'Minimum income adequacy threshold for loan eligibility.'),
    ('LTI Ratio ≤ 10×',           lambda d: d.get('Loan_To_Income_Ratio', 0) <= 10.0,
     lambda d: f"{d.get('Loan_To_Income_Ratio', 0):.1f}×",
     'Loan exposure must not exceed 10× annual income.'),
    ('Applicant Age ≤ 65',        lambda d: d.get('Age', 30) <= 65,
     lambda d: f"{d.get('Age', 30)} yrs",
     'Loan tenure must be serviceable within working/pensionable age.'),
]

def get_policy_audit(details):
    """
    Returns structured policy compliance results.
    Output: list of (rule_name, passed: bool, actual_value: str, rationale: str)
    """
    return [
        (rule, condition(details), value_fn(details), rationale)
        for rule, condition, value_fn, rationale in _POLICY_RULES
    ]


def find_fastest_approval_path(model_data, details, current_score, current_pd, current_grade):
    """
    Identifies top improvement actions that would advance the risk grade.
    Returns: (list of improvement dicts, target_grade: str)
    """
    from utils import calculate_emi

    grade_order = ['D', 'C', 'B', 'B+', 'A', 'A+']
    idx = grade_order.index(current_grade) if current_grade in grade_order else 0
    target_grade = grade_order[min(idx + 1, len(grade_order) - 1)]

    paths = []
    income = details.get('Monthly_Income', 1)

    # ── Action 1: Improve CIBIL ──────────────────────────────────────────────
    if details['CIBIL_Score'] < 850:
        for delta in [25, 50, 75, 100, 150]:
            test = details.copy()
            test['CIBIL_Score'] = min(details['CIBIL_Score'] + delta, 900)
            cs = calculate_5cs_scores(test)
            ns = calculate_weighted_score(cs)
            ng = get_risk_grade(ns)
            np_ = predict_single_probability(model_data, test)
            if ng != current_grade:
                paths.append({
                    'icon': '🎯', 'priority': 1,
                    'action': f'Improve CIBIL by {delta} points → {test["CIBIL_Score"]}',
                    'why': 'CIBIL is the #1 predictor of default. Even small improvements have outsized scoring impact.',
                    'new_score': ns, 'new_grade': ng, 'new_pd': np_,
                    'pd_delta': round((np_ - current_pd) * 100, 2),
                    'score_delta': round(ns - current_score, 1),
                })
                break

    # ── Action 2: Reduce loan amount ─────────────────────────────────────────
    if details['Loan_Amount'] > 100000:
        for pct in [0.10, 0.15, 0.20, 0.25, 0.30]:
            test = details.copy()
            reduced = max(round(details['Loan_Amount'] * (1 - pct) / 10000) * 10000, 50000)
            test['Loan_Amount'] = reduced
            test['Loan_To_Income_Ratio'] = reduced / (income * 12) if income > 0 else 5
            test['Net_Worth'] = (details.get('Savings', 0) + details.get('Investments', 0) +
                                 details.get('Asset_Value', 0) - reduced)
            cs = calculate_5cs_scores(test)
            ns = calculate_weighted_score(cs)
            ng = get_risk_grade(ns)
            np_ = predict_single_probability(model_data, test)
            if ng != current_grade:
                saved = details['Loan_Amount'] - reduced
                paths.append({
                    'icon': '💰', 'priority': 2,
                    'action': f'Reduce loan by ₹{saved:,.0f} → ₹{reduced:,.0f}',
                    'why': 'Lower loan amount reduces LTI ratio and loan concentration risk.',
                    'new_score': ns, 'new_grade': ng, 'new_pd': np_,
                    'pd_delta': round((np_ - current_pd) * 100, 2),
                    'score_delta': round(ns - current_score, 1),
                })
                break

    # ── Action 3: Build savings ───────────────────────────────────────────────
    if details['Savings'] < income * 6:
        for mult in [3, 4, 5, 6]:
            test = details.copy()
            test['Savings'] = income * mult
            test['Net_Worth'] = (income * mult + details.get('Investments', 0) +
                                 details.get('Asset_Value', 0) - details.get('Loan_Amount', 0))
            cs = calculate_5cs_scores(test)
            ns = calculate_weighted_score(cs)
            ng = get_risk_grade(ns)
            np_ = predict_single_probability(model_data, test)
            if ng != current_grade:
                paths.append({
                    'icon': '🏦', 'priority': 3,
                    'action': f'Build savings to ₹{test["Savings"]:,.0f} ({mult}× monthly income)',
                    'why': 'Stronger savings buffer reduces Loss-Given-Default and boosts Capital score.',
                    'new_score': ns, 'new_grade': ng, 'new_pd': np_,
                    'pd_delta': round((np_ - current_pd) * 100, 2),
                    'score_delta': round(ns - current_score, 1),
                })
                break

    # ── Action 4: Close existing loans (reduce EMI burden) ───────────────────
    if details.get('Existing_EMIs', 0) > income * 0.15:
        for reduction in [0.30, 0.50, 0.70]:
            test = details.copy()
            new_emis = round(details['Existing_EMIs'] * (1 - reduction), 0)
            test['Existing_EMIs'] = new_emis
            if income > 0:
                test['DTI_Ratio'] = round((new_emis / income) * 100, 1)
            cs = calculate_5cs_scores(test)
            ns = calculate_weighted_score(cs)
            ng = get_risk_grade(ns)
            np_ = predict_single_probability(model_data, test)
            if ng != current_grade:
                paths.append({
                    'icon': '🔄', 'priority': 4,
                    'action': f'Pre-close {int(reduction*100)}% of existing loans → EMI ₹{new_emis:,.0f}/m',
                    'why': 'Reducing EMI burden directly lowers DTI — the second most important risk factor.',
                    'new_score': ns, 'new_grade': ng, 'new_pd': np_,
                    'pd_delta': round((np_ - current_pd) * 100, 2),
                    'score_delta': round(ns - current_score, 1),
                })
                break

    return sorted(paths, key=lambda x: x['priority'])[:3], target_grade


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — UNIFIED APPLICATION PROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_single_application(details, model_data):
    """
    Full underwriting pipeline for a single application.
    Takes a raw borrower dict + trained model, returns comprehensive result dict.
    Used by: upload engine, individual assessment, AI assistant lookups.
    """
    from utils import calculate_emi

    # Ensure derived features
    income = details.get('Monthly_Income', 0)
    if 'Loan_To_Income_Ratio' not in details or details['Loan_To_Income_Ratio'] is None:
        details['Loan_To_Income_Ratio'] = (
            details.get('Loan_Amount', 0) / (income * 12) if income > 0 else 5.0
        )
    if 'Net_Worth' not in details or details['Net_Worth'] is None:
        details['Net_Worth'] = (
            details.get('Savings', 0) + details.get('Investments', 0) +
            details.get('Asset_Value', 0) - details.get('Loan_Amount', 0)
        )
    if 'Existing_EMIs' not in details:
        details['Existing_EMIs'] = 0
    if 'DTI_Ratio' not in details or details['DTI_Ratio'] is None:
        if income > 0:
            details['DTI_Ratio'] = round((details.get('Existing_EMIs', 0) / income) * 100, 1)
        else:
            details['DTI_Ratio'] = 0.0

    # 5 Cs Scoring
    cs = calculate_5cs_scores(details)
    w_score = calculate_weighted_score(cs)
    grade = get_risk_grade(w_score)
    g_label = get_risk_grade_label(grade)

    # Policy checks
    stops = check_hard_stops(details)
    flags, fc = check_red_flags(details)
    verdict, risk_cat, detail_msg = get_rule_based_decision(w_score, stops, fc)

    # ML prediction
    ml_pd = predict_single_probability(model_data, details)
    ml_conf = get_confidence_score(model_data, details)

    # Consensus
    agree, consensus_exp, final_verdict, decision_source = get_model_consensus(
        verdict, ml_pd, w_score
    )

    # Generate explanation
    explanation = detail_msg
    if stops:
        explanation = f"REJECTED — {stops[0]}"
    elif not agree:
        explanation = f"{final_verdict}: {consensus_exp[:120]}"

    return {
        'cs': cs,
        'w_score': w_score,
        'grade': grade,
        'g_label': g_label,
        'stops': stops,
        'flags': flags,
        'fc': fc,
        'verdict': verdict,
        'risk_cat': risk_cat,
        'detail_msg': detail_msg,
        'ml_pd': ml_pd,
        'ml_conf': ml_conf,
        'agree': agree,
        'consensus_exp': consensus_exp,
        'final_verdict': final_verdict,
        'decision_source': decision_source,
        'explanation': explanation,
    }


def process_batch_applications(app_list, model_data):
    """
    Process a list of application dicts through the full underwriting pipeline.
    Returns list of result dicts (one per application).
    """
    results = []
    for details in app_list:
        try:
            result = process_single_application(details.copy(), model_data)
            result['details'] = details
            results.append(result)
        except Exception as e:
            results.append({
                'details': details,
                'final_verdict': 'Error',
                'explanation': str(e),
                'w_score': 0, 'ml_pd': 0, 'grade': 'D',
            })
    return results


if __name__ == '__main__':
    csv_path   = os.path.join(BASE_DIR, 'data', 'borrowers.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'risk_model.pkl')
    if os.path.exists(csv_path):
        train_ml_model(csv_path, model_path)
    else:
        print("Run generator.py first.")


