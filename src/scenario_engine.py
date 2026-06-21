"""
scenario_engine.py — Portfolio-Level Macro-Economic Stress Testing Engine
═════════════════════════════════════════════════════════════════════════════
Simulates 5 macro-economic scenarios across a portfolio sample and measures
the impact on default probability, credit scores, risk grades and approval
rates. Designed for CreditIQ's portfolio analytics dashboard.

Scenarios are aligned with RBI's ICAAP / NBFC stress-testing guidelines and
cover GDP slowdown, recession, rate shocks, inflation and employment crises.
═════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
from model import (
    calculate_5cs_scores, calculate_weighted_score,
    check_hard_stops, check_red_flags, get_rule_based_decision,
    get_risk_grade, predict_single_probability,
)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = {
    'Economic Slowdown': {
        'description': 'GDP growth declines to 4-5%, moderate income compression across sectors',
        'icon': '📉',
        'severity': 'Moderate',
        'shocks': {
            'income_reduction': 0.15,
            'savings_reduction': 0.20,
        },
    },
    'Recession': {
        'description': 'Severe economic contraction, mass layoffs, liquidity crisis',
        'icon': '🔴',
        'severity': 'Critical',
        'shocks': {
            'income_reduction': 0.30,
            'savings_reduction': 0.40,
            'missed_emi_increase': 2,
        },
    },
    'Interest Rate Increase': {
        'description': 'RBI aggressive monetary tightening, repo rate increase by 300bps',
        'icon': '📈',
        'severity': 'Severe',
        'shocks': {
            'emi_burden_increase': 0.25,
        },
    },
    'Inflation Shock': {
        'description': 'CPI inflation exceeds 8%, real income erosion, cost of living spike',
        'icon': '🔥',
        'severity': 'Severe',
        'shocks': {
            'income_reduction': 0.20,
            'savings_reduction': 0.25,
        },
    },
    'Employment Crisis': {
        'description': 'Sector-wide layoffs, gig economy disruption, IT/services downturn',
        'icon': '🏭',
        'severity': 'Severe',
        'shocks': {
            'employment_halved': True,
            'income_reduction': 0.20,
            'missed_emi_increase': 1,
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_scenario_names() -> list:
    """Returns list of available scenario names."""
    return ['Economic Slowdown', 'Recession', 'Interest Rate Increase',
            'Inflation Shock', 'Employment Crisis']


def _grade_rank(grade: str) -> int:
    """Maps a risk grade to an ordinal rank (higher = better)."""
    return {'A+': 6, 'A': 5, 'B+': 4, 'B': 3, 'C': 2, 'D': 1}.get(grade, 0)


def _risk_category_from_grade(grade: str) -> str:
    """Returns a broad risk bucket label for a grade."""
    return {
        'A+': 'Very Low Risk', 'A': 'Low Risk', 'B+': 'Moderate Risk',
        'B': 'Elevated Risk', 'C': 'High Risk', 'D': 'Very High Risk',
    }.get(grade, 'Unknown')


def _is_high_risk(grade: str) -> bool:
    """Returns True if the grade falls in the High / Very High risk bucket."""
    return grade in ('C', 'D')


def _is_approved(verdict: str) -> bool:
    """Returns True for approval-family verdicts."""
    return verdict in ('Approved', 'Approved with Conditions')


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO SHOCK APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def _apply_shocks(record: dict, shocks: dict) -> dict:
    """
    Applies macro-economic shocks to a single applicant record (dict).
    Returns a new dict with stressed field values and recalculated DTI.
    """
    stressed = record.copy()

    # ── Income reduction ──────────────────────────────────────────────────
    if 'income_reduction' in shocks:
        factor = 1.0 - shocks['income_reduction']
        stressed['Monthly_Income'] = round(record['Monthly_Income'] * factor, 0)

    # ── Savings reduction ─────────────────────────────────────────────────
    if 'savings_reduction' in shocks:
        factor = 1.0 - shocks['savings_reduction']
        stressed['Savings'] = round(record['Savings'] * factor, 0)

    # ── Missed EMI increase ───────────────────────────────────────────────
    if 'missed_emi_increase' in shocks:
        stressed['Missed_EMIs'] = int(record['Missed_EMIs'] + shocks['missed_emi_increase'])

    # ── EMI burden increase (rate hike scenario) ──────────────────────────
    if 'emi_burden_increase' in shocks:
        factor = 1.0 + shocks['emi_burden_increase']
        stressed['Existing_EMIs'] = round(record['Existing_EMIs'] * factor, 0)

    # ── Employment stability halved ───────────────────────────────────────
    if shocks.get('employment_halved'):
        stressed['Employment_Length'] = round(record['Employment_Length'] / 2.0, 1)

    # ── Recalculate DTI after income / EMI changes ────────────────────────
    income = stressed['Monthly_Income']
    if income > 0:
        stressed['DTI_Ratio'] = round(
            (stressed['Existing_EMIs'] / income) * 100, 1
        )
    else:
        stressed['DTI_Ratio'] = 100.0  # Cap at 100% for zero-income edge case

    return stressed


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-RECORD EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_record(record: dict, model_data: dict) -> dict:
    """
    Runs the full credit evaluation pipeline on a single record dict.
    Returns: {score, grade, pd, verdict, risk_category}
    """
    cs_scores = calculate_5cs_scores(record)
    weighted  = calculate_weighted_score(cs_scores)
    grade     = get_risk_grade(weighted)
    pd_value  = predict_single_probability(model_data, record)
    hard_stops = check_hard_stops(record)
    _flags, flag_count = check_red_flags(record)
    verdict, risk_cat, _ = get_rule_based_decision(weighted, hard_stops, flag_count)

    return {
        'score': weighted,
        'grade': grade,
        'pd': pd_value,
        'verdict': verdict,
        'risk_category': risk_cat,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLE_MIN = 500
_SAMPLE_MAX = 1000


def _sample_portfolio(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Draws a random sample from the portfolio for stress-test performance.
    Min 500, max 1000 records. If portfolio is smaller than 500, uses all rows.
    """
    n = len(portfolio_df)
    if n <= _SAMPLE_MIN:
        return portfolio_df.copy()
    sample_size = min(n, _SAMPLE_MAX)
    return portfolio_df.sample(n=sample_size, random_state=42).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE — PORTFOLIO STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_portfolio_stress_test(
    portfolio_df: pd.DataFrame,
    model_data: dict,
    scenario_name: str = 'all',
) -> list:
    """
    Runs stress test on portfolio sample. Returns list of scenario result dicts.

    Parameters
    ──────────
    portfolio_df : pd.DataFrame
        Full portfolio DataFrame with standard CreditIQ fields.
    model_data : dict
        Trained model artefact from model.train_ml_model().
    scenario_name : str
        Name of a single scenario to run, or 'all' (default) to run all five.

    Returns
    ───────
    list[dict]  — One result dict per scenario (see module docstring for schema).
    """
    # ── Determine which scenarios to run ──────────────────────────────────
    if scenario_name == 'all':
        scenarios_to_run = SCENARIOS
    else:
        if scenario_name not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {get_scenario_names()}"
            )
        scenarios_to_run = {scenario_name: SCENARIOS[scenario_name]}

    # ── Sample the portfolio ──────────────────────────────────────────────
    sample_df = _sample_portfolio(portfolio_df)
    records = sample_df.to_dict(orient='records')
    sample_size = len(records)

    # ── Evaluate BASE (unstressed) portfolio ──────────────────────────────
    base_results = [_evaluate_record(rec, model_data) for rec in records]
    base_pds      = [r['pd'] for r in base_results]
    base_scores   = [r['score'] for r in base_results]
    base_grades   = [r['grade'] for r in base_results]
    base_verdicts = [r['verdict'] for r in base_results]

    base_avg_pd        = round(float(np.mean(base_pds)), 4)
    base_avg_score     = round(float(np.mean(base_scores)), 1)
    base_approval_rate = round(
        sum(1 for v in base_verdicts if _is_approved(v)) / sample_size * 100, 1
    )
    base_high_risk_count = sum(1 for g in base_grades if _is_high_risk(g))

    # ── Run each scenario ─────────────────────────────────────────────────
    results = []

    for name, config in scenarios_to_run.items():
        shocks = config['shocks']

        # Apply shocks to every sampled record
        stressed_records = [_apply_shocks(rec, shocks) for rec in records]

        # Evaluate stressed portfolio
        stressed_results  = [_evaluate_record(sr, model_data) for sr in stressed_records]
        stressed_pds      = [r['pd'] for r in stressed_results]
        stressed_scores   = [r['score'] for r in stressed_results]
        stressed_grades   = [r['grade'] for r in stressed_results]
        stressed_verdicts = [r['verdict'] for r in stressed_results]

        # ── Aggregate metrics ─────────────────────────────────────────────
        stressed_avg_pd = round(float(np.mean(stressed_pds)), 4)
        stressed_avg_score = round(float(np.mean(stressed_scores)), 1)
        stressed_approval_rate = round(
            sum(1 for v in stressed_verdicts if _is_approved(v)) / sample_size * 100, 1
        )
        stressed_high_risk_count = sum(
            1 for g in stressed_grades if _is_high_risk(g)
        )

        # ── Grade migration ───────────────────────────────────────────────
        upgrades = 0
        downgrades = 0
        stable = 0
        for bg, sg in zip(base_grades, stressed_grades):
            rank_diff = _grade_rank(sg) - _grade_rank(bg)
            if rank_diff > 0:
                upgrades += 1
            elif rank_diff < 0:
                downgrades += 1
            else:
                stable += 1

        # ── High risk increase (percentage-point change) ──────────────────
        base_hr_pct = round(base_high_risk_count / sample_size * 100, 1)
        stressed_hr_pct = round(stressed_high_risk_count / sample_size * 100, 1)
        high_risk_increase = round(stressed_hr_pct - base_hr_pct, 1)

        results.append({
            'scenario':              name,
            'description':           config['description'],
            'icon':                  config['icon'],
            'base_avg_pd':           base_avg_pd,
            'stressed_avg_pd':       stressed_avg_pd,
            'pd_change':             round(stressed_avg_pd - base_avg_pd, 4),
            'base_approval_rate':    base_approval_rate,
            'stressed_approval_rate': stressed_approval_rate,
            'approval_change':       round(stressed_approval_rate - base_approval_rate, 1),
            'base_avg_score':        base_avg_score,
            'stressed_avg_score':    stressed_avg_score,
            'score_change':          round(stressed_avg_score - base_avg_score, 1),
            'grade_migration': {
                'upgrades':   upgrades,
                'downgrades': downgrades,
                'stable':     stable,
            },
            'high_risk_increase':    high_risk_increase,
            'severity':              config['severity'],
        })

    return results
