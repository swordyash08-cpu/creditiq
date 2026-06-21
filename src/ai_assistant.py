"""
CreditIQ — AI Credit Assistant (Rule-Based NLP Engine)
═══════════════════════════════════════════════════════════════════════════════
Self-contained, intelligent Credit Risk Officer chatbot.
Detects user intent via keyword matching and regex, queries the SQLite
database, and returns formatted, contextual responses.

No external API required — runs entirely offline.

Supported intents:
  1. Individual Application Lookup
  2. Risk Factor Analysis
  3. Portfolio Count Queries
  4. Segment Analysis
  5. Threshold Queries
  6. Portfolio Metrics
  7. Risk Grade Explanation
  8. Policy Queries
  9. General Credit Knowledge
═══════════════════════════════════════════════════════════════════════════════
"""

import re


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INTENT DETECTION PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

# Regex to extract applicant IDs from queries
_ID_PATTERNS = [
    re.compile(r'(SYN-\d{4,5})', re.IGNORECASE),
    re.compile(r'(APP-\d{4,5})', re.IGNORECASE),
    re.compile(r'(?:application|app|applicant|borrower|id)\s*#?\s*(\d{3,5})', re.IGNORECASE),
]

# Threshold extraction patterns
_THRESHOLD_PATTERNS = {
    'pd_above':    re.compile(r'(?:pd|default\s*probability|probability)\s*(?:above|over|greater|>|exceeds?)\s*([\d.]+)\s*%?', re.IGNORECASE),
    'pd_below':    re.compile(r'(?:pd|default\s*probability|probability)\s*(?:below|under|less|<)\s*([\d.]+)\s*%?', re.IGNORECASE),
    'cibil_above': re.compile(r'(?:cibil|credit\s*score)\s*(?:above|over|greater|>)\s*(\d+)', re.IGNORECASE),
    'cibil_below': re.compile(r'(?:cibil|credit\s*score)\s*(?:below|under|less|<)\s*(\d+)', re.IGNORECASE),
    'dti_above':   re.compile(r'(?:dti|debt.to.income)\s*(?:above|over|greater|>|exceeds?)\s*([\d.]+)\s*%?', re.IGNORECASE),
    'dti_below':   re.compile(r'(?:dti|debt.to.income)\s*(?:below|under|less|<)\s*([\d.]+)\s*%?', re.IGNORECASE),
    'income_above': re.compile(r'(?:income|salary)\s*(?:above|over|greater|>)\s*(\d[\d,]*)', re.IGNORECASE),
    'income_below': re.compile(r'(?:income|salary)\s*(?:below|under|less|<)\s*(\d[\d,]*)', re.IGNORECASE),
}

# Intent keyword groups
_INTENT_KEYWORDS = {
    'application_lookup': [
        'application', 'applicant', 'borrower', 'app-', 'syn-',
        'tell me about', 'look up', 'lookup', 'what happened to',
        'why was', 'status of', 'details of', 'show me',
    ],
    'risk_factors': [
        'risk factor', 'risk driver', 'contributed most', 'top factor',
        'biggest risk', 'main risk', 'key risk', 'what factors',
        'major driver', 'risk contributor',
    ],
    'count_query': [
        'how many', 'count of', 'number of', 'total number', 'count',
    ],
    'segment_analysis': [
        'segment', 'compare', 'comparison', 'by category', 'by employment',
        'which category', 'which segment', 'group by', 'breakdown',
        'default probability by', 'highest default', 'lowest default',
    ],
    'portfolio_metrics': [
        'approval rate', 'average credit', 'average cibil', 'portfolio summary',
        'average income', 'total aum', 'total loan', 'portfolio stat',
        'portfolio overview', 'avg pd', 'average pd', 'mean score',
        'portfolio health', 'overall', 'summary',
    ],
    'risk_grade_explain': [
        'why is', 'instead of', 'risk grade', 'grade explanation',
        'scoring threshold', 'how is the grade', 'grade calculated',
        'risk classification', 'what makes', 'medium risk', 'high risk',
        'low risk', 'grade system',
    ],
    'policy_query': [
        'policy', 'credit policy', 'minimum cibil', 'maximum dti',
        'policy rule', 'threshold', 'eligibility', 'cut-off',
        'cutoff', 'underwriting rule', 'lending criteria',
    ],
    'general_knowledge': [
        'what is dti', 'what is pd', 'explain pd', 'explain dti',
        'what is cibil', 'what are the 5 cs', '5 cs of credit',
        'what is lgd', 'what is ead', 'what is risk score',
        'define', 'explain', 'meaning of', 'what does',
        'how does scoring work', 'credit scoring',
    ],
}

# ── Credit knowledge base ────────────────────────────────────────────────────

_KNOWLEDGE_BASE = {
    'dti': (
        "📊 **Debt-to-Income (DTI) Ratio**\n\n"
        "DTI measures how much of a borrower's monthly income goes towards "
        "servicing existing debt obligations.\n\n"
        "**Formula:** `DTI = (Total Monthly EMIs ÷ Monthly Income) × 100`\n\n"
        "| DTI Range | Risk Level | Interpretation |\n"
        "|-----------|------------|----------------|\n"
        "| ≤ 30%     | ✅ Low     | Comfortable repayment capacity |\n"
        "| 31–50%    | ⚠️ Medium  | Manageable but limited headroom |\n"
        "| > 50%     | 🔴 High    | Strained — high default probability |\n\n"
        "Most Indian banks and NBFCs cap DTI at 50–60% as per RBI guidelines."
    ),
    'pd': (
        "📊 **Probability of Default (PD)**\n\n"
        "PD estimates the likelihood that a borrower will fail to meet their "
        "debt obligations within a specified period (typically 12 months).\n\n"
        "**CreditIQ uses two engines to estimate PD:**\n"
        "1. **Rule-Based Scorecard** — 5 Cs weighted scoring (0–100)\n"
        "2. **Random Forest ML Model** — trained on historical default data\n\n"
        "| PD Range   | Risk Category | Typical Action |\n"
        "|------------|---------------|----------------|\n"
        "| < 10%      | ✅ Low Risk    | Auto-approve at best rate |\n"
        "| 10–25%     | ⚠️ Medium     | Approve with conditions |\n"
        "| 25–50%     | 🟠 High       | Manual review required |\n"
        "| > 50%      | 🔴 Very High  | Reject / decline |\n\n"
        "PD is a core input to Expected Loss: `EL = PD × LGD × EAD`."
    ),
    'cibil': (
        "📊 **CIBIL Score**\n\n"
        "CIBIL (Credit Information Bureau India Limited) score ranges from "
        "300 to 900. It is India's primary credit score, similar to FICO.\n\n"
        "| Score Range | Rating       | Loan Eligibility |\n"
        "|-------------|--------------|------------------|\n"
        "| 800–900     | 🌟 Excellent | Best rates, instant approval |\n"
        "| 750–799     | ✅ Good      | Standard approval |\n"
        "| 700–749     | ⚠️ Fair      | Conditional approval |\n"
        "| 650–699     | 🟠 Below Avg | Higher rates, co-applicant needed |\n"
        "| < 650       | 🔴 Poor      | Likely rejection |\n\n"
        "CreditIQ uses a minimum CIBIL threshold of 600 as a hard stop. "
        "Scores are sourced from TransUnion CIBIL."
    ),
    '5cs': (
        "📊 **The 5 Cs of Credit**\n\n"
        "The foundational framework used by lenders worldwide to assess "
        "creditworthiness:\n\n"
        "1. **Character (35%)** — Credit history, CIBIL score, missed payments. "
        "Indicates willingness to repay.\n\n"
        "2. **Capacity (30%)** — Income, DTI ratio, employment stability. "
        "Measures ability to service the loan.\n\n"
        "3. **Capital (15%)** — Savings, investments, net worth. "
        "Represents the borrower's financial cushion.\n\n"
        "4. **Collateral (15%)** — Asset value, LTV ratio. "
        "Security available in case of default.\n\n"
        "5. **Conditions (5%)** — Loan purpose, tenure, market conditions. "
        "External factors affecting repayment.\n\n"
        "CreditIQ's scorecard assigns a weighted score on a 0–100 scale "
        "using exactly this framework."
    ),
    'lgd': (
        "📊 **Loss Given Default (LGD)**\n\n"
        "LGD represents the percentage of exposure a lender expects to lose "
        "if a borrower defaults, after accounting for recoveries.\n\n"
        "**Formula:** `LGD = 1 − Recovery Rate`\n\n"
        "Typical LGD ranges:\n"
        "• Secured (home loans): 20–35%\n"
        "• Unsecured (personal loans): 60–85%\n"
        "• Credit cards: 75–95%\n\n"
        "LGD is a key component of Expected Loss: `EL = PD × LGD × EAD`."
    ),
    'risk_score': (
        "📊 **CreditIQ Risk Score (0–100)**\n\n"
        "The composite risk score combines all 5 Cs into a single number:\n\n"
        "| Grade | Score Range | Risk Level |\n"
        "|-------|-------------|------------|\n"
        "| A+    | ≥ 88        | Prime — Lowest Risk |\n"
        "| A     | 80–87       | Near-Prime — Low Risk |\n"
        "| B+    | 72–79       | Standard — Moderate Risk |\n"
        "| B     | 62–71       | Sub-Standard — Elevated Risk |\n"
        "| C     | 50–61       | Speculative — High Risk |\n"
        "| D     | < 50        | Distressed — Very High Risk |\n\n"
        "Decisions: A+/A → Approved, B+ → Approved with Conditions, "
        "B/C → Manual Review, D → Rejected."
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _get_db(db_module):
    """Resolves the database module — uses injected module or imports directly."""
    if db_module is not None:
        return db_module
    import database as db
    return db


def _extract_applicant_id(query):
    """
    Extracts an applicant ID from the query string.
    Supports formats: SYN-01007, APP-00123, or bare number 1007.
    Returns normalised ID string or None.
    """
    for pattern in _ID_PATTERNS:
        match = pattern.search(query)
        if match:
            raw = match.group(1)
            # If it's already prefixed, normalise the case and padding
            if raw.upper().startswith('SYN-') or raw.upper().startswith('APP-'):
                prefix = raw[:4].upper()
                num = raw[4:].lstrip('0') or '0'
                return f"{prefix}{int(num):05d}"
            # Bare number → default to SYN- prefix
            return f"SYN-{int(raw):05d}"
    return None


def _fmt_inr(amount):
    """Compact Indian Rupee formatter."""
    if amount is None or amount == 0:
        return "₹0"
    if abs(amount) >= 1_00_00_000:
        return f"₹{amount / 1_00_00_000:.2f} Cr"
    if abs(amount) >= 1_00_000:
        return f"₹{amount / 1_00_000:.2f} L"
    return f"₹{amount:,.0f}"


def _detect_intent(query):
    """
    Scores each intent category by counting keyword matches.
    Returns the intent with the highest score, or 'unknown'.
    """
    q = query.lower()
    scores = {}
    for intent, keywords in _INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[intent] = score

    # Special boost: if an applicant ID is found, heavily favour lookup
    if _extract_applicant_id(query):
        scores['application_lookup'] = scores.get('application_lookup', 0) + 10

    # Threshold patterns override to threshold intent
    for key, pattern in _THRESHOLD_PATTERNS.items():
        if pattern.search(query):
            return 'threshold_query'

    if not scores:
        return 'unknown'
    return max(scores, key=scores.get)


def _decision_emoji(decision):
    """Returns an emoji for the decision type."""
    if not decision:
        return "❓"
    d = decision.lower()
    if 'rejected' in d:
        return "🔴"
    if 'manual' in d:
        return "🟡"
    if 'condition' in d:
        return "🟠"
    return "✅"


def _grade_emoji(grade):
    """Returns an emoji for the risk grade."""
    mapping = {'A+': '🌟', 'A': '✅', 'B+': '⚠️', 'B': '🟠', 'C': '🔴', 'D': '⛔'}
    return mapping.get(grade, '❓')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RESPONSE GENERATORS (one per intent)
# ══════════════════════════════════════════════════════════════════════════════

def _handle_application_lookup(query, db):
    """Looks up a specific application and explains the decision."""
    app_id = _extract_applicant_id(query)
    if not app_id:
        return (
            "⚠️ **Application ID Not Found**\n\n"
            "I couldn't extract a valid application ID from your query.\n\n"
            "**Supported formats:**\n"
            "• `SYN-01007` or `APP-00123`\n"
            "• \"Tell me about application 1007\"\n"
            "• \"Why was 01007 rejected?\""
        )

    # Try with extracted ID first; if not found and prefix was APP, try SYN
    app = db.get_application(app_id)
    if not app and app_id.startswith('APP-'):
        alt_id = 'SYN-' + app_id[4:]
        app = db.get_application(alt_id)
        if app:
            app_id = alt_id

    if not app:
        return (
            f"⚠️ **Application `{app_id}` Not Found**\n\n"
            f"No record exists for this application ID in the database. "
            f"Please verify the ID and try again."
        )

    # Build response
    decision = app.get('decision', 'N/A')
    grade = app.get('risk_grade', 'N/A')
    score = app.get('risk_score', 0)
    pd_val = app.get('pd_value', 0) or 0
    cibil = app.get('cibil_score', 0)
    dti = app.get('dti_ratio', 0)
    income = app.get('monthly_income', 0)
    loan_amt = app.get('loan_amount', 0)
    missed = app.get('missed_emis', 0)
    explanation = app.get('decision_explanation', '')
    category = app.get('category', 'N/A')
    net_worth = app.get('net_worth', 0)

    lines = [
        f"{_decision_emoji(decision)} **Application: `{app_id}`**\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| 📋 Category | {category} |",
        f"| 🏦 CIBIL Score | {cibil} |",
        f"| 📊 DTI Ratio | {dti:.1f}% |",
        f"| 💰 Monthly Income | {_fmt_inr(income)} |",
        f"| 🏷️ Loan Amount | {_fmt_inr(loan_amt)} |",
        f"| 💳 Missed EMIs | {missed} |",
        f"| 💎 Net Worth | {_fmt_inr(net_worth)} |",
        f"| 🎯 Risk Score | {score:.1f} / 100 |",
        f"| 📈 PD (Default Prob.) | {pd_val * 100:.1f}% |",
        f"| {_grade_emoji(grade)} Risk Grade | **{grade}** |",
        f"| {_decision_emoji(decision)} Decision | **{decision}** |\n",
    ]

    # Decision rationale
    lines.append("**📝 Decision Rationale:**\n")
    if explanation:
        lines.append(f"> {explanation}\n")

    # Additional risk commentary
    risk_flags = []
    if cibil < 650:
        risk_flags.append(f"• 🔴 CIBIL score ({cibil}) is below the 650 threshold — indicates poor repayment history")
    if dti > 50:
        risk_flags.append(f"• 🔴 DTI ({dti:.1f}%) exceeds 50% — strained repayment capacity")
    if missed >= 3:
        risk_flags.append(f"• 🔴 {missed} missed EMIs — signals active delinquency")
    if income < 20000:
        risk_flags.append(f"• ⚠️ Monthly income ({_fmt_inr(income)}) is on the lower end")
    if pd_val > 0.25:
        risk_flags.append(f"• 🔴 PD of {pd_val*100:.1f}% indicates elevated default risk")

    if risk_flags:
        lines.append("**⚠️ Key Risk Factors Identified:**\n")
        lines.extend(risk_flags)
    else:
        lines.append("✅ No significant risk flags detected for this application.")

    return "\n".join(lines)


def _handle_risk_factors(query, db):
    """Analyses top risk drivers across the portfolio."""
    try:
        df = db.get_portfolio_df()
    except Exception:
        return "⚠️ Unable to load portfolio data. Please ensure the database is initialised."

    if df.empty:
        return "📭 The portfolio is empty. No risk factors to analyse."

    n = len(df)
    lines = ["📊 **Portfolio Risk Factor Analysis**\n"]

    # CIBIL distribution
    low_cibil = len(df[df['cibil_score'] < 650])
    lines.append(f"**1. CIBIL Score Distribution**")
    lines.append(f"   • Mean: {df['cibil_score'].mean():.0f} | Median: {df['cibil_score'].median():.0f}")
    lines.append(f"   • Below 650 (high risk): **{low_cibil}** ({low_cibil/n*100:.1f}%)\n")

    # DTI
    high_dti = len(df[df['dti_ratio'] > 50])
    lines.append(f"**2. Debt-to-Income Ratio**")
    lines.append(f"   • Mean DTI: {df['dti_ratio'].mean():.1f}%")
    lines.append(f"   • DTI > 50% (strained): **{high_dti}** ({high_dti/n*100:.1f}%)\n")

    # Missed EMIs
    delinquent = len(df[df['missed_emis'] >= 2])
    lines.append(f"**3. Payment Delinquency**")
    lines.append(f"   • 2+ missed EMIs: **{delinquent}** ({delinquent/n*100:.1f}%)\n")

    # PD distribution
    if 'pd_value' in df.columns:
        high_pd = len(df[df['pd_value'] > 0.25])
        lines.append(f"**4. Default Probability**")
        lines.append(f"   • Mean PD: {df['pd_value'].mean()*100:.1f}%")
        lines.append(f"   • PD > 25%: **{high_pd}** ({high_pd/n*100:.1f}%)\n")

    # Income adequacy
    low_income = len(df[df['monthly_income'] < 20000])
    lines.append(f"**5. Income Adequacy**")
    lines.append(f"   • Income < ₹20,000: **{low_income}** ({low_income/n*100:.1f}%)\n")

    # Top risk driver summary
    lines.append("**🎯 Top Risk Drivers (ranked by impact):**")
    risk_scores = [
        ('Low CIBIL Score (< 650)', low_cibil / n * 100),
        ('High DTI (> 50%)', high_dti / n * 100),
        ('Payment Delinquency (2+ missed)', delinquent / n * 100),
        ('Low Income (< ₹20K)', low_income / n * 100),
    ]
    risk_scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (driver, pct) in enumerate(risk_scores, 1):
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        lines.append(f"   {rank}. {driver}: {pct:.1f}% `{bar}`")

    return "\n".join(lines)


def _handle_count_query(query, db):
    """Handles 'how many' / 'count' type queries."""
    q = query.lower()
    stats = db.get_portfolio_stats()
    total = stats.get('total', 0)

    if total == 0:
        return "📭 The portfolio is empty. No applications to count."

    decisions = stats.get('decisions', {})
    grades = stats.get('grades', {})

    # Count by decision
    if 'rejected' in q or 'reject' in q:
        cnt = decisions.get('Rejected', 0)
        return f"🔴 **Rejected Applications: {cnt}** out of {total} ({cnt/total*100:.1f}%)"
    if 'approved' in q and 'condition' not in q:
        cnt = sum(v for k, v in decisions.items() if k and 'Approved' in k)
        return f"✅ **Approved Applications: {cnt}** out of {total} ({cnt/total*100:.1f}%)"
    if 'condition' in q:
        cnt = decisions.get('Approved with Conditions', 0)
        return f"🟠 **Conditionally Approved: {cnt}** out of {total} ({cnt/total*100:.1f}%)"
    if 'manual' in q or 'review' in q:
        cnt = decisions.get('Manual Review', 0)
        return f"🟡 **Manual Review Applications: {cnt}** out of {total} ({cnt/total*100:.1f}%)"

    # Count by risk level
    if 'high risk' in q or 'high-risk' in q:
        cnt = grades.get('C', 0) + grades.get('D', 0)
        return f"🔴 **High Risk Borrowers (Grade C + D): {cnt}** out of {total} ({cnt/total*100:.1f}%)"
    if 'low risk' in q or 'low-risk' in q:
        cnt = grades.get('A+', 0) + grades.get('A', 0)
        return f"✅ **Low Risk Borrowers (Grade A+ and A): {cnt}** out of {total} ({cnt/total*100:.1f}%)"
    if 'medium risk' in q or 'moderate risk' in q:
        cnt = grades.get('B+', 0) + grades.get('B', 0)
        return f"⚠️ **Medium Risk Borrowers (Grade B+ and B): {cnt}** out of {total} ({cnt/total*100:.1f}%)"

    # Count by grade
    for grade_key in ['A+', 'A', 'B+', 'B', 'C', 'D']:
        if grade_key.lower() in q or f"grade {grade_key.lower()}" in q:
            cnt = grades.get(grade_key, 0)
            return f"{_grade_emoji(grade_key)} **Grade {grade_key} Applications: {cnt}** out of {total} ({cnt/total*100:.1f}%)"

    # Default: total count with breakdown
    lines = [f"📊 **Total Applications: {total}**\n"]
    lines.append("**By Decision:**")
    for dec, cnt in sorted(decisions.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"   {_decision_emoji(dec)} {dec}: {cnt} ({cnt/total*100:.1f}%)")
    lines.append("\n**By Risk Grade:**")
    for g in ['A+', 'A', 'B+', 'B', 'C', 'D']:
        cnt = grades.get(g, 0)
        if cnt > 0:
            lines.append(f"   {_grade_emoji(g)} Grade {g}: {cnt} ({cnt/total*100:.1f}%)")

    return "\n".join(lines)


def _handle_segment_analysis(query, db):
    """Groups portfolio by category/segment and shows risk stats."""
    try:
        df = db.get_portfolio_df()
    except Exception:
        return "⚠️ Unable to load portfolio data."

    if df.empty:
        return "📭 The portfolio is empty."

    lines = ["📊 **Segment-wise Risk Analysis**\n"]
    lines.append("| Segment | Count | Avg CIBIL | Avg DTI | Avg PD | Avg Score | Reject % |")
    lines.append("|---------|-------|-----------|---------|--------|-----------|----------|")

    for cat in sorted(df['category'].dropna().unique()):
        seg = df[df['category'] == cat]
        n = len(seg)
        avg_cibil = seg['cibil_score'].mean()
        avg_dti = seg['dti_ratio'].mean()
        avg_pd = seg['pd_value'].mean() * 100 if 'pd_value' in seg.columns else 0
        avg_score = seg['risk_score'].mean() if 'risk_score' in seg.columns else 0
        rej = len(seg[seg['decision'] == 'Rejected'])
        rej_pct = rej / n * 100 if n > 0 else 0
        lines.append(
            f"| {cat} | {n} | {avg_cibil:.0f} | {avg_dti:.1f}% | {avg_pd:.1f}% | {avg_score:.1f} | {rej_pct:.1f}% |"
        )

    # Identify highest default segment
    seg_pd = df.groupby('category')['pd_value'].mean()
    if not seg_pd.empty:
        worst = seg_pd.idxmax()
        lines.append(f"\n🎯 **Highest default probability segment: `{worst}`** "
                     f"with mean PD of {seg_pd[worst]*100:.1f}%")

    return "\n".join(lines)


def _handle_threshold_query(query, db):
    """Filters borrowers by numeric thresholds (PD, CIBIL, DTI, Income)."""
    filters = {}
    limit = 10
    description = "matching borrowers"

    for key, pattern in _THRESHOLD_PATTERNS.items():
        match = pattern.search(query)
        if match:
            val = float(match.group(1).replace(',', ''))
            if key == 'pd_above':
                filters['min_pd'] = val / 100 if val > 1 else val
                description = f"PD above {val}%"
            elif key == 'pd_below':
                filters['max_pd'] = val / 100 if val > 1 else val
                description = f"PD below {val}%"
            elif key == 'cibil_above':
                filters['min_cibil'] = int(val)
                description = f"CIBIL above {int(val)}"
            elif key == 'cibil_below':
                filters['max_cibil'] = int(val)
                description = f"CIBIL below {int(val)}"
            elif key == 'dti_above':
                filters['min_pd'] = None  # clear
                description = f"DTI above {val}%"
            elif key == 'dti_below':
                description = f"DTI below {val}%"
            elif key == 'income_above':
                filters['min_income'] = val
                description = f"income above {_fmt_inr(val)}"
            elif key == 'income_below':
                filters['max_income'] = val
                description = f"income below {_fmt_inr(val)}"

    filters['limit'] = limit
    try:
        result_df = db.search_applications(filters)
    except Exception:
        return "⚠️ Unable to query the database."

    # For DTI filters (not directly supported in search_applications), filter locally
    if 'dti_above' in query.lower() or 'dti_below' in query.lower():
        try:
            df = db.get_portfolio_df()
            for key, pattern in _THRESHOLD_PATTERNS.items():
                match = pattern.search(query)
                if match:
                    val = float(match.group(1).replace(',', ''))
                    if key == 'dti_above':
                        result_df = df[df['dti_ratio'] > val].head(limit)
                    elif key == 'dti_below':
                        result_df = df[df['dti_ratio'] < val].head(limit)
        except Exception:
            pass

    if result_df.empty:
        return f"📭 No applicants found with {description}."

    total_matching = len(result_df)
    lines = [f"📊 **Applicants with {description}** (showing top {min(total_matching, limit)})\n"]
    lines.append("| ID | CIBIL | DTI | Income | PD | Grade | Decision |")
    lines.append("|----|-------|-----|--------|------|-------|----------|")

    for _, row in result_df.head(limit).iterrows():
        app_id = row.get('applicant_id', 'N/A')
        cibil = row.get('cibil_score', 0)
        dti = row.get('dti_ratio', 0)
        income = row.get('monthly_income', 0)
        pd_v = row.get('pd_value', 0) or 0
        grade = row.get('risk_grade', 'N/A')
        dec = row.get('decision', 'N/A')
        lines.append(
            f"| {app_id} | {cibil} | {dti:.1f}% | {_fmt_inr(income)} | "
            f"{pd_v*100:.1f}% | {grade} | {dec} |"
        )

    return "\n".join(lines)


def _handle_portfolio_metrics(query, db):
    """Returns aggregate portfolio statistics."""
    stats = db.get_portfolio_stats()
    total = stats.get('total', 0)

    if total == 0:
        return "📭 The portfolio is empty. No metrics available."

    q = query.lower()

    # Specific metric shortcuts
    if 'approval rate' in q:
        rate = stats.get('approval_rate', 0)
        return f"✅ **Current Approval Rate: {rate:.1f}%** across {total} applications."

    if 'average cibil' in q or 'avg cibil' in q or 'average credit' in q:
        return f"🏦 **Average CIBIL Score: {stats.get('avg_cibil', 0):.0f}** across {total} applications."

    if 'average pd' in q or 'avg pd' in q:
        return f"📈 **Average PD: {stats.get('avg_pd', 0)*100:.1f}%** across {total} applications."

    if 'total aum' in q or 'total loan' in q:
        return f"💰 **Total AUM (Assets Under Management): {_fmt_inr(stats.get('total_aum', 0))}**"

    # Full summary
    decisions = stats.get('decisions', {})
    grades = stats.get('grades', {})

    lines = [
        "📊 **Portfolio Summary**\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| 📋 Total Applications | {total} |",
        f"| 🏦 Avg CIBIL Score | {stats.get('avg_cibil', 0):.0f} |",
        f"| 📊 Avg DTI Ratio | {stats.get('avg_dti', 0):.1f}% |",
        f"| 📈 Avg Default Probability | {stats.get('avg_pd', 0)*100:.1f}% |",
        f"| 🎯 Avg Risk Score | {stats.get('avg_risk_score', 0):.1f} / 100 |",
        f"| 💰 Total AUM | {_fmt_inr(stats.get('total_aum', 0))} |",
        f"| ✅ Approval Rate | {stats.get('approval_rate', 0):.1f}% |\n",
    ]

    lines.append("**Decision Breakdown:**")
    for dec in ['Approved', 'Approved with Conditions', 'Manual Review', 'Rejected']:
        cnt = decisions.get(dec, 0)
        if cnt > 0:
            pct = cnt / total * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"   {_decision_emoji(dec)} {dec}: {cnt} ({pct:.1f}%) `{bar}`")

    lines.append("\n**Grade Distribution:**")
    for g in ['A+', 'A', 'B+', 'B', 'C', 'D']:
        cnt = grades.get(g, 0)
        if cnt > 0:
            pct = cnt / total * 100
            lines.append(f"   {_grade_emoji(g)} Grade {g}: {cnt} ({pct:.1f}%)")

    return "\n".join(lines)


def _handle_risk_grade_explain(query, db):
    """Explains the risk grading system and scoring thresholds."""
    lines = [
        "📊 **Risk Grade Classification System**\n",
        "CreditIQ uses a dual-engine approach: a **rule-based 5C scorecard** and "
        "a **Random Forest ML model** to classify applicants.\n",
        "**Risk Grade Thresholds:**\n",
        "| Grade | Score Range | Risk Level | Typical Decision |",
        "|-------|-------------|------------|------------------|",
        "| 🌟 A+ | ≥ 88 | Prime — Lowest Risk | Auto-Approved at best rate |",
        "| ✅ A  | 80–87 | Near-Prime — Low Risk | Approved — standard terms |",
        "| ⚠️ B+ | 72–79 | Standard — Moderate Risk | Approved with Conditions |",
        "| 🟠 B  | 62–71 | Sub-Standard — Elevated Risk | Manual Review / Conditional |",
        "| 🔴 C  | 50–61 | Speculative — High Risk | Manual Review required |",
        "| ⛔ D  | < 50 | Distressed — Very High Risk | Rejected |",
        "",
        "**What determines the grade?**\n",
        "The composite score (0–100) is built from five weighted dimensions:\n",
        "1. **Character (35%)** — CIBIL score, missed EMIs, credit history length",
        "2. **Capacity (30%)** — DTI ratio, monthly income, employment tenure",
        "3. **Capital (15%)** — Savings-to-income ratio, investment portfolio",
        "4. **Collateral (15%)** — Loan-to-value ratio based on asset coverage",
        "5. **Conditions (5%)** — Loan purpose, tenure, loan-to-income ratio\n",
        "**Why might someone be Medium Risk instead of Low Risk?**\n",
        "Common reasons include:",
        "• CIBIL score in the 650–700 range (drags Character score)",
        "• DTI ratio above 40% (reduces Capacity score)",
        "• Thin credit history (< 3 years)",
        "• Low savings buffer relative to income",
        "• Missed EMIs signalling past delinquency",
        "• High loan-to-income ratio",
        "",
        "Additionally, the ML model may detect non-linear risk patterns (e.g., "
        "combination of moderate DTI + short employment + high LTI) that push "
        "the final PD above the low-risk threshold.",
    ]

    return "\n".join(lines)


def _handle_policy_query(query, db):
    """Returns current credit policy settings."""
    try:
        policies = db.get_policies()
    except Exception:
        policies = {}

    if not policies:
        return (
            "⚠️ No credit policies found in the database. "
            "Please initialise the system with default policies."
        )

    lines = [
        "📋 **Current Credit Policy Rules**\n",
        "| Policy | Value | Description |",
        "|--------|-------|-------------|",
    ]

    # Format display names and values nicely
    display_map = {
        'min_cibil_score':     ('Min CIBIL Score', lambda v: f"{v:.0f}"),
        'max_dti_ratio':       ('Max DTI Ratio', lambda v: f"{v:.0f}%"),
        'min_monthly_income':  ('Min Monthly Income', lambda v: _fmt_inr(v)),
        'min_savings':         ('Min Savings', lambda v: _fmt_inr(v)),
        'max_missed_emis':     ('Max Missed EMIs', lambda v: f"{v:.0f}"),
        'max_lti_ratio':       ('Max Loan-to-Income Ratio', lambda v: f"{v:.0f}×"),
        'pd_low_threshold':    ('PD Low Risk Threshold', lambda v: f"{v*100:.0f}%"),
        'pd_medium_threshold': ('PD Medium Risk Threshold', lambda v: f"{v*100:.0f}%"),
        'pd_high_threshold':   ('PD High Risk Threshold', lambda v: f"{v*100:.0f}%"),
        'score_approve':       ('Auto-Approval Score', lambda v: f"{v:.0f} / 100"),
        'score_conditional':   ('Conditional Approval Score', lambda v: f"{v:.0f} / 100"),
        'score_manual':        ('Manual Review Score', lambda v: f"{v:.0f} / 100"),
    }

    for name, info in policies.items():
        val = info.get('value', 0)
        desc = info.get('desc', '')
        if name in display_map:
            label, fmt_fn = display_map[name]
            lines.append(f"| {label} | **{fmt_fn(val)}** | {desc} |")
        else:
            lines.append(f"| {name} | **{val}** | {desc} |")

    lines.append("\n💡 *Policies are configurable via the Settings panel. "
                 "Changes take effect immediately for new applications.*")

    return "\n".join(lines)


def _handle_general_knowledge(query):
    """Returns educational content about credit concepts."""
    q = query.lower()

    # Match specific topics
    if 'dti' in q or 'debt to income' in q or 'debt-to-income' in q:
        return _KNOWLEDGE_BASE['dti']
    if 'pd' in q and ('what' in q or 'explain' in q or 'define' in q or 'meaning' in q):
        return _KNOWLEDGE_BASE['pd']
    if 'cibil' in q:
        return _KNOWLEDGE_BASE['cibil']
    if '5 c' in q or 'five c' in q or '5c' in q:
        return _KNOWLEDGE_BASE['5cs']
    if 'lgd' in q or 'loss given' in q:
        return _KNOWLEDGE_BASE['lgd']
    if 'risk score' in q or 'scoring' in q or 'how does' in q:
        return _KNOWLEDGE_BASE['risk_score']

    # Generic credit explanation
    return (
        "📚 **Credit Risk Concepts**\n\n"
        "I can explain the following topics — just ask!\n\n"
        "• **DTI** — Debt-to-Income ratio\n"
        "• **PD** — Probability of Default\n"
        "• **CIBIL** — India's primary credit score\n"
        "• **5 Cs of Credit** — Character, Capacity, Capital, Collateral, Conditions\n"
        "• **LGD** — Loss Given Default\n"
        "• **Risk Score** — How CreditIQ computes the 0–100 composite score\n\n"
        "Try asking: *\"What is DTI?\"* or *\"Explain the 5 Cs of credit.\"*"
    )


def _handle_unknown(query):
    """Fallback response listing available capabilities."""
    return (
        "🤖 **CreditIQ Assistant — How Can I Help?**\n\n"
        "I didn't quite understand that query. Here's what I can do:\n\n"
        "**📋 Application Lookup**\n"
        "   → *\"Tell me about application SYN-01007\"*\n"
        "   → *\"Why was APP-00123 rejected?\"*\n\n"
        "**📊 Risk Analysis**\n"
        "   → *\"What are the top risk factors in the portfolio?\"*\n"
        "   → *\"Which segment has the highest default probability?\"*\n\n"
        "**🔢 Counts & Metrics**\n"
        "   → *\"How many high risk borrowers exist?\"*\n"
        "   → *\"What is the approval rate?\"*\n"
        "   → *\"Portfolio summary\"*\n\n"
        "**🔍 Threshold Queries**\n"
        "   → *\"Show applicants with PD above 20%\"*\n"
        "   → *\"List borrowers with CIBIL below 600\"*\n\n"
        "**📖 Policy & Knowledge**\n"
        "   → *\"What are the credit policy rules?\"*\n"
        "   → *\"What is DTI?\"*\n"
        "   → *\"Explain the 5 Cs of credit\"*\n\n"
        "**📈 Risk Grades**\n"
        "   → *\"Explain the risk grade system\"*\n"
        "   → *\"Why is someone Medium Risk instead of Low Risk?\"*"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def process_query(query: str, db_module=None) -> str:
    """
    Main entry point for the AI Credit Assistant.

    Detects user intent via keyword matching and regex, queries the
    SQLite database, and returns intelligent, formatted responses.

    Args:
        query:     Natural language question from the user.
        db_module: Optional database module for dependency injection.
                   If None, imports `database` directly.

    Returns:
        Formatted response string with metrics, rationale, and emojis.
    """
    if not query or not query.strip():
        return "💬 Please enter a question about the credit portfolio."

    query = query.strip()
    intent = _detect_intent(query)

    try:
        if intent == 'application_lookup':
            db = _get_db(db_module)
            return _handle_application_lookup(query, db)

        elif intent == 'risk_factors':
            db = _get_db(db_module)
            return _handle_risk_factors(query, db)

        elif intent == 'count_query':
            db = _get_db(db_module)
            return _handle_count_query(query, db)

        elif intent == 'segment_analysis':
            db = _get_db(db_module)
            return _handle_segment_analysis(query, db)

        elif intent == 'threshold_query':
            db = _get_db(db_module)
            return _handle_threshold_query(query, db)

        elif intent == 'portfolio_metrics':
            db = _get_db(db_module)
            return _handle_portfolio_metrics(query, db)

        elif intent == 'risk_grade_explain':
            db = _get_db(db_module)
            return _handle_risk_grade_explain(query, db)

        elif intent == 'policy_query':
            db = _get_db(db_module)
            return _handle_policy_query(query, db)

        elif intent == 'general_knowledge':
            return _handle_general_knowledge(query)

        else:
            return _handle_unknown(query)

    except Exception as e:
        return (
            f"⚠️ **Error Processing Query**\n\n"
            f"Something went wrong while processing your request:\n"
            f"`{type(e).__name__}: {str(e)}`\n\n"
            f"Please try rephrasing your question or check that the "
            f"database is initialised."
        )
