"""
CreditIQ — SQLite Persistence Layer
Stores applications, upload history, chat logs, and credit policies.
Data persists across Streamlit restarts.
"""

import sqlite3
import os
import json
import datetime
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'creditiq.db')


def _get_conn():
    """Returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_database():
    """Creates all tables if they don't exist."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS applications (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        applicant_id          TEXT UNIQUE,
        category              TEXT,
        age                   INTEGER,
        monthly_income        REAL,
        savings               REAL,
        investments           REAL,
        cibil_score           INTEGER,
        missed_emis           INTEGER,
        credit_history_length INTEGER,
        existing_emis         REAL,
        dti_ratio             REAL,
        employment_length     REAL,
        asset_value           REAL,
        loan_amount           REAL,
        loan_tenure           INTEGER,
        loan_purpose          TEXT,
        co_applicant          INTEGER DEFAULT 0,
        loan_to_income_ratio  REAL,
        net_worth             REAL,
        risk_score            REAL,
        pd_value              REAL,
        risk_grade            TEXT,
        decision              TEXT,
        decision_explanation  TEXT DEFAULT '',
        source                TEXT DEFAULT 'synthetic',
        upload_batch_id       TEXT DEFAULT '',
        created_at            TEXT DEFAULT (datetime('now')),
        updated_at            TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS upload_history (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id          TEXT UNIQUE,
        filename          TEXT,
        total_records     INTEGER,
        accepted          INTEGER,
        rejected          INTEGER,
        data_quality      REAL,
        created_at        TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS chat_logs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        role        TEXT,
        content     TEXT,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS credit_policies (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        policy_name     TEXT,
        policy_value    REAL,
        description     TEXT,
        updated_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_app_risk_grade ON applications(risk_grade);
    CREATE INDEX IF NOT EXISTS idx_app_decision ON applications(decision);
    CREATE INDEX IF NOT EXISTS idx_app_cibil ON applications(cibil_score);
    CREATE INDEX IF NOT EXISTS idx_app_source ON applications(source);
    """)

    conn.commit()
    conn.close()


def seed_from_csv(csv_path, model_data=None):
    """Seeds the database with existing borrower CSV data on first run."""
    conn = _get_conn()
    cursor = conn.cursor()

    # Check if already seeded
    count = cursor.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    if count > 0:
        conn.close()
        return count

    if not os.path.exists(csv_path):
        conn.close()
        return 0

    df = pd.read_csv(csv_path)

    # Import scoring functions
    from model import (
        calculate_5cs_scores, calculate_weighted_score,
        check_hard_stops, check_red_flags, get_rule_based_decision,
        get_risk_grade, predict_single_probability,
        get_model_consensus,
    )

    records = []
    for idx, row in df.iterrows():
        d = row.to_dict()

        # Ensure derived features
        if 'Loan_To_Income_Ratio' not in d:
            inc = d.get('Monthly_Income', 1)
            d['Loan_To_Income_Ratio'] = d.get('Loan_Amount', 0) / (inc * 12) if inc > 0 else 5.0
        if 'Net_Worth' not in d:
            d['Net_Worth'] = (d.get('Savings', 0) + d.get('Investments', 0) +
                              d.get('Asset_Value', 0) - d.get('Loan_Amount', 0))

        # Calculate risk metrics
        cs = calculate_5cs_scores(d)
        w_score = calculate_weighted_score(cs)
        grade = get_risk_grade(w_score)
        stops = check_hard_stops(d)
        _, fc = check_red_flags(d)
        verdict, risk_cat, detail_msg = get_rule_based_decision(w_score, stops, fc)

        pd_val = 0.0
        if model_data:
            try:
                pd_val = predict_single_probability(model_data, d)
                _, _, final_verdict, _ = get_model_consensus(verdict, pd_val, w_score)
                verdict = final_verdict
            except Exception:
                pass

        app_id = f"SYN-{idx + 1:05d}"

        records.append((
            app_id,
            d.get('Category', 'Salaried'),
            int(d.get('Age', 30)),
            float(d.get('Monthly_Income', 0)),
            float(d.get('Savings', 0)),
            float(d.get('Investments', 0)),
            int(d.get('CIBIL_Score', 650)),
            int(d.get('Missed_EMIs', 0)),
            int(d.get('Credit_History_Length', 0)),
            float(d.get('Existing_EMIs', 0)),
            float(d.get('DTI_Ratio', 0)),
            float(d.get('Employment_Length', 0)),
            float(d.get('Asset_Value', 0)),
            float(d.get('Loan_Amount', 0)),
            int(d.get('Loan_Tenure', 36)),
            d.get('Loan_Purpose', 'Personal'),
            int(d.get('Co_Applicant', 0)),
            float(d.get('Loan_To_Income_Ratio', 0)),
            float(d.get('Net_Worth', 0)),
            round(w_score, 2),
            round(pd_val, 4),
            grade,
            verdict,
            detail_msg,
            'synthetic',
            'SEED',
        ))

    cursor.executemany("""
        INSERT OR IGNORE INTO applications (
            applicant_id, category, age, monthly_income, savings, investments,
            cibil_score, missed_emis, credit_history_length, existing_emis,
            dti_ratio, employment_length, asset_value, loan_amount, loan_tenure,
            loan_purpose, co_applicant, loan_to_income_ratio, net_worth,
            risk_score, pd_value, risk_grade, decision, decision_explanation,
            source, upload_batch_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, records)

    conn.commit()
    inserted = cursor.rowcount
    conn.close()
    return len(records)


# ── CRUD Operations ──────────────────────────────────────────────────────────

def insert_application(app_dict):
    """Inserts a single application record. Returns the applicant_id."""
    conn = _get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO applications (
            applicant_id, category, age, monthly_income, savings, investments,
            cibil_score, missed_emis, credit_history_length, existing_emis,
            dti_ratio, employment_length, asset_value, loan_amount, loan_tenure,
            loan_purpose, co_applicant, loan_to_income_ratio, net_worth,
            risk_score, pd_value, risk_grade, decision, decision_explanation,
            source, upload_batch_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        app_dict.get('applicant_id', ''),
        app_dict.get('category', ''),
        app_dict.get('age', 0),
        app_dict.get('monthly_income', 0),
        app_dict.get('savings', 0),
        app_dict.get('investments', 0),
        app_dict.get('cibil_score', 0),
        app_dict.get('missed_emis', 0),
        app_dict.get('credit_history_length', 0),
        app_dict.get('existing_emis', 0),
        app_dict.get('dti_ratio', 0),
        app_dict.get('employment_length', 0),
        app_dict.get('asset_value', 0),
        app_dict.get('loan_amount', 0),
        app_dict.get('loan_tenure', 0),
        app_dict.get('loan_purpose', ''),
        app_dict.get('co_applicant', 0),
        app_dict.get('loan_to_income_ratio', 0),
        app_dict.get('net_worth', 0),
        app_dict.get('risk_score', 0),
        app_dict.get('pd_value', 0),
        app_dict.get('risk_grade', ''),
        app_dict.get('decision', ''),
        app_dict.get('decision_explanation', ''),
        app_dict.get('source', 'manual'),
        app_dict.get('upload_batch_id', ''),
    ))

    conn.commit()
    conn.close()
    return app_dict.get('applicant_id', '')


def insert_applications_batch(app_list):
    """Inserts multiple application records in a batch."""
    conn = _get_conn()
    cursor = conn.cursor()

    records = []
    for app_dict in app_list:
        records.append((
            app_dict.get('applicant_id', ''),
            app_dict.get('category', ''),
            app_dict.get('age', 0),
            app_dict.get('monthly_income', 0),
            app_dict.get('savings', 0),
            app_dict.get('investments', 0),
            app_dict.get('cibil_score', 0),
            app_dict.get('missed_emis', 0),
            app_dict.get('credit_history_length', 0),
            app_dict.get('existing_emis', 0),
            app_dict.get('dti_ratio', 0),
            app_dict.get('employment_length', 0),
            app_dict.get('asset_value', 0),
            app_dict.get('loan_amount', 0),
            app_dict.get('loan_tenure', 0),
            app_dict.get('loan_purpose', ''),
            app_dict.get('co_applicant', 0),
            app_dict.get('loan_to_income_ratio', 0),
            app_dict.get('net_worth', 0),
            app_dict.get('risk_score', 0),
            app_dict.get('pd_value', 0),
            app_dict.get('risk_grade', ''),
            app_dict.get('decision', ''),
            app_dict.get('decision_explanation', ''),
            app_dict.get('source', 'upload'),
            app_dict.get('upload_batch_id', ''),
        ))

    cursor.executemany("""
        INSERT OR REPLACE INTO applications (
            applicant_id, category, age, monthly_income, savings, investments,
            cibil_score, missed_emis, credit_history_length, existing_emis,
            dti_ratio, employment_length, asset_value, loan_amount, loan_tenure,
            loan_purpose, co_applicant, loan_to_income_ratio, net_worth,
            risk_score, pd_value, risk_grade, decision, decision_explanation,
            source, upload_batch_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, records)

    conn.commit()
    conn.close()
    return len(records)


def get_application(applicant_id):
    """Returns a single application as a dict, or None."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM applications WHERE applicant_id = ?", (applicant_id,)
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_portfolio_df():
    """Returns the full portfolio as a pandas DataFrame with original CamelCase column names."""
    conn = _get_conn()
    df = pd.read_sql_query("SELECT * FROM applications ORDER BY id", conn)
    conn.close()
    
    # Map back to CamelCase for model.py compatibility
    col_map = {
        'applicant_id': 'Applicant_ID',
        'category': 'Category',
        'age': 'Age',
        'monthly_income': 'Monthly_Income',
        'savings': 'Savings',
        'investments': 'Investments',
        'cibil_score': 'CIBIL_Score',
        'missed_emis': 'Missed_EMIs',
        'credit_history_length': 'Credit_History_Length',
        'existing_emis': 'Existing_EMIs',
        'dti_ratio': 'DTI_Ratio',
        'employment_length': 'Employment_Length',
        'asset_value': 'Asset_Value',
        'loan_amount': 'Loan_Amount',
        'loan_tenure': 'Loan_Tenure',
        'loan_purpose': 'Loan_Purpose',
        'co_applicant': 'Co_Applicant',
        'loan_to_income_ratio': 'Loan_To_Income_Ratio',
        'net_worth': 'Net_Worth',
        'risk_score': 'Risk_Score',
        'pd_value': 'PD_Value',
        'risk_grade': 'Risk_Grade',
        'decision': 'Decision',
        'decision_explanation': 'Decision_Explanation',
        'source': 'Source',
        'upload_batch_id': 'Upload_Batch_ID',
    }
    df.rename(columns=col_map, inplace=True)
    return df


def get_portfolio_stats():
    """Returns aggregate portfolio statistics."""
    conn = _get_conn()
    cursor = conn.cursor()

    stats = {}
    stats['total'] = cursor.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    stats['avg_cibil'] = cursor.execute("SELECT AVG(cibil_score) FROM applications").fetchone()[0] or 0
    stats['avg_pd'] = cursor.execute("SELECT AVG(pd_value) FROM applications").fetchone()[0] or 0
    stats['avg_risk_score'] = cursor.execute("SELECT AVG(risk_score) FROM applications").fetchone()[0] or 0
    stats['avg_dti'] = cursor.execute("SELECT AVG(dti_ratio) FROM applications").fetchone()[0] or 0
    stats['total_aum'] = cursor.execute("SELECT SUM(loan_amount) FROM applications").fetchone()[0] or 0

    # Decision breakdown
    decisions = cursor.execute(
        "SELECT decision, COUNT(*) as cnt FROM applications GROUP BY decision"
    ).fetchall()
    stats['decisions'] = {row['decision']: row['cnt'] for row in decisions}

    approved = sum(v for k, v in stats['decisions'].items()
                   if k and ('Approved' in k))
    stats['approval_rate'] = (approved / stats['total'] * 100) if stats['total'] > 0 else 0

    # Grade distribution
    grades = cursor.execute(
        "SELECT risk_grade, COUNT(*) as cnt FROM applications GROUP BY risk_grade"
    ).fetchall()
    stats['grades'] = {row['risk_grade']: row['cnt'] for row in grades}

    conn.close()
    return stats


def search_applications(filters=None):
    """
    Search applications with multiple filters.
    filters dict keys: applicant_id, risk_grade, decision,
                       min_pd, max_pd, min_income, max_income,
                       min_cibil, max_cibil, category, limit
    """
    conn = _get_conn()
    query = "SELECT * FROM applications WHERE 1=1"
    params = []

    if filters:
        if filters.get('applicant_id'):
            query += " AND applicant_id LIKE ?"
            params.append(f"%{filters['applicant_id']}%")
        if filters.get('risk_grade'):
            if isinstance(filters['risk_grade'], list):
                placeholders = ','.join(['?'] * len(filters['risk_grade']))
                query += f" AND risk_grade IN ({placeholders})"
                params.extend(filters['risk_grade'])
            else:
                query += " AND risk_grade = ?"
                params.append(filters['risk_grade'])
        if filters.get('decision'):
            if isinstance(filters['decision'], list):
                placeholders = ','.join(['?'] * len(filters['decision']))
                query += f" AND decision IN ({placeholders})"
                params.extend(filters['decision'])
            else:
                query += " AND decision LIKE ?"
                params.append(f"%{filters['decision']}%")
        if filters.get('min_pd') is not None:
            query += " AND pd_value >= ?"
            params.append(filters['min_pd'])
        if filters.get('max_pd') is not None:
            query += " AND pd_value <= ?"
            params.append(filters['max_pd'])
        if filters.get('min_income') is not None:
            query += " AND monthly_income >= ?"
            params.append(filters['min_income'])
        if filters.get('max_income') is not None:
            query += " AND monthly_income <= ?"
            params.append(filters['max_income'])
        if filters.get('min_cibil') is not None:
            query += " AND cibil_score >= ?"
            params.append(filters['min_cibil'])
        if filters.get('max_cibil') is not None:
            query += " AND cibil_score <= ?"
            params.append(filters['max_cibil'])
        if filters.get('category'):
            query += " AND category = ?"
            params.append(filters['category'])

    query += " ORDER BY id DESC"

    if filters and filters.get('limit'):
        query += " LIMIT ?"
        params.append(filters['limit'])
    else:
        query += " LIMIT 500"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


# ── Upload History ────────────────────────────────────────────────────────────

def log_upload(batch_id, filename, total, accepted, rejected, quality):
    """Logs an upload batch."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO upload_history (batch_id, filename, total_records, accepted, rejected, data_quality)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (batch_id, filename, total, accepted, rejected, quality))
    conn.commit()
    conn.close()


def get_upload_history():
    """Returns upload history as DataFrame."""
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM upload_history ORDER BY created_at DESC", conn
    )
    conn.close()
    return df


# ── Chat Logs ─────────────────────────────────────────────────────────────────

def log_chat(role, content):
    """Saves a chat message."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO chat_logs (role, content) VALUES (?, ?)",
        (role, content)
    )
    conn.commit()
    conn.close()


def get_chat_history(limit=50):
    """Returns recent chat history."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT role, content, created_at FROM chat_logs ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]
    except Exception:
        return []


def clear_chat_history():
    """Clears all chat logs."""
    conn = _get_conn()
    conn.execute("DELETE FROM chat_logs")
    conn.commit()
    conn.close()


# ── Credit Policies ───────────────────────────────────────────────────────────

DEFAULT_POLICIES = {
    'min_cibil_score':    {'value': 650,   'desc': 'Minimum CIBIL score for eligibility'},
    'max_dti_ratio':      {'value': 60.0,  'desc': 'Maximum post-loan DTI ratio (%)'},
    'min_monthly_income': {'value': 15000, 'desc': 'Minimum monthly income (₹)'},
    'min_savings':        {'value': 20000, 'desc': 'Minimum savings balance (₹)'},
    'max_missed_emis':    {'value': 3,     'desc': 'Maximum missed EMIs allowed'},
    'max_lti_ratio':      {'value': 10.0,  'desc': 'Maximum loan-to-income ratio'},
    'pd_low_threshold':   {'value': 0.15,  'desc': 'PD threshold: Low Risk upper bound'},
    'pd_medium_threshold': {'value': 0.30, 'desc': 'PD threshold: Medium Risk upper bound'},
    'pd_high_threshold':  {'value': 0.50,  'desc': 'PD threshold: High Risk upper bound'},
    'score_approve':      {'value': 72,    'desc': 'Minimum score for auto-approval'},
    'score_conditional':  {'value': 60,    'desc': 'Minimum score for conditional approval'},
    'score_manual':       {'value': 50,    'desc': 'Minimum score for manual review'},
}


def init_default_policies():
    """Initializes default credit policies if not already set."""
    conn = _get_conn()
    cursor = conn.cursor()
    existing = cursor.execute("SELECT COUNT(*) FROM credit_policies").fetchone()[0]
    if existing == 0:
        for name, info in DEFAULT_POLICIES.items():
            cursor.execute(
                "INSERT INTO credit_policies (policy_name, policy_value, description) VALUES (?, ?, ?)",
                (name, info['value'], info['desc'])
            )
        conn.commit()
    conn.close()


def get_policies():
    """Returns all credit policies as a dict."""
    conn = _get_conn()
    rows = conn.execute("SELECT policy_name, policy_value, description FROM credit_policies").fetchall()
    conn.close()
    return {row['policy_name']: {'value': row['policy_value'], 'desc': row['description']} for row in rows}


def update_policy(name, value):
    """Updates a single policy value."""
    conn = _get_conn()
    conn.execute(
        "UPDATE credit_policies SET policy_value = ?, updated_at = datetime('now') WHERE policy_name = ?",
        (value, name)
    )
    conn.commit()
    conn.close()


def reset_policies():
    """Resets all policies to defaults."""
    conn = _get_conn()
    conn.execute("DELETE FROM credit_policies")
    conn.commit()
    conn.close()
    init_default_policies()


def get_next_applicant_id():
    """Returns the next sequential applicant ID."""
    conn = _get_conn()
    cursor = conn.cursor()
    count = cursor.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    conn.close()
    return f"APP-{count + 1:05d}"
