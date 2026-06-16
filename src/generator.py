import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_borrower_dataset(num_records=8000, seed=42):
    """
    Generates a realistic synthetic dataset for credit risk assessment
    aligned with Indian retail lending contexts.
    Borrower segments: Student, Salaried, Self-Employed, Retired
    Target default rate: ~18% (realistic Indian NBFC/Bank portfolio)
    """
    np.random.seed(seed)

    categories = ['Student', 'Salaried', 'Self-Employed', 'Retired']
    category_choices = np.random.choice(
        categories, size=num_records, p=[0.15, 0.50, 0.25, 0.10]
    )

    data = []

    for cat in category_choices:
        co_applicant = 0

        if cat == 'Student':
            age = int(np.random.randint(18, 26))
            income = float(np.random.choice(
                [0, np.random.randint(5000, 15000)], p=[0.65, 0.35]
            ))
            savings = float(np.random.randint(1000, 60000))
            investments = float(np.random.randint(0, 25000))
            cibil = int(np.random.choice(
                [np.random.randint(650, 760), np.random.randint(580, 650)],
                p=[0.55, 0.45]
            ))
            missed_emis = int(np.random.choice([0, 1, 2], p=[0.88, 0.09, 0.03]))
            credit_hist_len = int(np.random.randint(0, 4))
            existing_emis = float(np.random.choice(
                [0, np.random.randint(1000, 5000)], p=[0.78, 0.22]
            ))
            emp_length = 0.0
            asset_value = float(np.random.choice(
                [0, np.random.randint(10000, 250000)], p=[0.88, 0.12]
            ))
            loan_amount = float(np.random.randint(30000, 600000))
            tenure_months = int(np.random.choice([12, 24, 36, 60, 84]))
            loan_purpose = np.random.choice(['Education', 'Personal'], p=[0.82, 0.18])
            co_applicant = int(np.random.choice([1, 0], p=[0.80, 0.20]))

        elif cat == 'Salaried':
            age = int(np.random.randint(22, 58))
            income = float(np.random.randint(20000, 200000))
            savings = float(np.random.randint(10000, 900000))
            investments = float(np.random.randint(0, 600000))
            cibil = int(np.random.randint(580, 860))
            missed_emis = int(np.random.choice(
                [0, 1, np.random.randint(2, 6)], p=[0.73, 0.17, 0.10]
            ))
            credit_hist_len = int(np.random.randint(1, 16))
            existing_emis = float(np.random.choice(
                [0, np.random.randint(5000, 45000)], p=[0.35, 0.65]
            ))
            if existing_emis > income * 0.70:
                existing_emis = round(income * np.random.uniform(0.10, 0.50))
            emp_length = float(round(np.random.uniform(0.5, 14.0), 1))
            asset_value = float(np.random.choice(
                [0, np.random.randint(500000, 6000000)], p=[0.58, 0.42]
            ))
            loan_amount = float(np.random.randint(50000, 3000000))
            tenure_months = int(np.random.choice([12, 24, 36, 60, 120, 180]))
            loan_purpose = np.random.choice(
                ['Home Purchase', 'Personal', 'Medical', 'Education'],
                p=[0.40, 0.30, 0.20, 0.10]
            )
            co_applicant = int(np.random.choice([1, 0], p=[0.30, 0.70]))

        elif cat == 'Self-Employed':
            age = int(np.random.randint(25, 60))
            income = float(np.random.randint(30000, 280000))
            savings = float(np.random.randint(20000, 1800000))
            investments = float(np.random.randint(0, 1200000))
            cibil = int(np.random.randint(540, 840))
            missed_emis = int(np.random.choice(
                [0, 1, np.random.randint(2, 7)], p=[0.62, 0.20, 0.18]
            ))
            credit_hist_len = int(np.random.randint(2, 22))
            existing_emis = float(np.random.choice(
                [0, np.random.randint(8000, 70000)], p=[0.28, 0.72]
            ))
            if existing_emis > income * 0.70:
                existing_emis = round(income * np.random.uniform(0.10, 0.50))
            emp_length = float(round(np.random.uniform(1.0, 18.0), 1))
            asset_value = float(np.random.choice(
                [0, np.random.randint(1000000, 9000000)], p=[0.35, 0.65]
            ))
            loan_amount = float(np.random.randint(100000, 5000000))
            tenure_months = int(np.random.choice([12, 24, 36, 60, 120]))
            loan_purpose = np.random.choice(
                ['Business Expansion', 'Home Purchase', 'Personal', 'Medical'],
                p=[0.50, 0.20, 0.15, 0.15]
            )
            co_applicant = int(np.random.choice([1, 0], p=[0.40, 0.60]))

        else:  # Retired
            age = int(np.random.randint(60, 76))
            income = float(np.random.randint(15000, 90000))
            savings = float(np.random.randint(100000, 3000000))
            investments = float(np.random.randint(50000, 2000000))
            cibil = int(np.random.randint(620, 830))
            missed_emis = int(np.random.choice([0, 1, 2], p=[0.84, 0.13, 0.03]))
            credit_hist_len = int(np.random.randint(10, 36))
            existing_emis = float(np.random.choice(
                [0, np.random.randint(3000, 18000)], p=[0.68, 0.32]
            ))
            if existing_emis > income * 0.70:
                existing_emis = round(income * np.random.uniform(0.10, 0.30))
            emp_length = 0.0
            asset_value = float(np.random.randint(500000, 5000000))
            loan_amount = float(np.random.randint(30000, 600000))
            tenure_months = int(np.random.choice([12, 24, 36, 60]))
            loan_purpose = np.random.choice(['Medical', 'Personal'], p=[0.60, 0.40])
            co_applicant = int(np.random.choice([1, 0], p=[0.20, 0.80]))

        # ── Derived features ──────────────────────────────────────────────────
        # DTI: Debt-to-Income Ratio
        if income > 0:
            dti = round((existing_emis / income) * 100, 1)
        else:
            dti = 100.0 if existing_emis > 0 else 0.0
        dti = min(dti, 100.0)

        # Loan-to-Income Ratio (annualised)
        annual_income = income * 12
        loan_to_income = round(loan_amount / annual_income, 2) if annual_income > 0 else 10.0
        loan_to_income = min(loan_to_income, 20.0)

        # Net Worth proxy
        net_worth = savings + investments + asset_value - loan_amount
        net_worth = max(net_worth, -loan_amount)

        # ── Realistic default probability simulation ───────────────────────────
        risk_score = 0.0

        # CIBIL impact (strongest signal)
        if cibil < 600:
            risk_score += 0.55
        elif cibil < 650:
            risk_score += 0.38
        elif cibil < 700:
            risk_score += 0.20
        elif cibil < 750:
            risk_score += 0.05
        else:
            risk_score -= 0.18

        # DTI impact
        if dti > 65:
            risk_score += 0.45
        elif dti > 50:
            risk_score += 0.28
        elif dti > 40:
            risk_score += 0.12
        elif dti < 25:
            risk_score -= 0.12

        # Missed EMIs (strong recency signal)
        risk_score += missed_emis * 0.18

        # Savings cushion
        if annual_income > 0:
            savings_ratio = savings / annual_income
            if savings_ratio < 0.05:
                risk_score += 0.18
            elif savings_ratio < 0.10:
                risk_score += 0.08
            elif savings_ratio > 0.30:
                risk_score -= 0.12

        # Loan-to-Income impact
        if loan_to_income > 6:
            risk_score += 0.20
        elif loan_to_income > 4:
            risk_score += 0.10
        elif loan_to_income < 2:
            risk_score -= 0.05

        # Employment stability
        if cat == 'Salaried':
            if emp_length < 1:
                risk_score += 0.18
            elif emp_length > 5:
                risk_score -= 0.10
        elif cat == 'Self-Employed':
            if emp_length < 2:
                risk_score += 0.22
            elif emp_length > 7:
                risk_score -= 0.08

        # Segment-specific adjustments
        if cat == 'Student' and not co_applicant:
            risk_score += 0.28
        if cat == 'Retired' and age > 70:
            risk_score += 0.10
        if cat == 'Self-Employed':
            risk_score += 0.05  # Inherent income volatility

        # Collateral coverage cushion
        if asset_value > 0 and loan_amount > 0:
            ltv = loan_amount / (asset_value + 1)
            if ltv < 0.50:
                risk_score -= 0.10
            elif ltv > 1.0:
                risk_score += 0.10

        # Sigmoid conversion → probability of default (~18% base rate)
        pd_val = 1.0 / (1.0 + np.exp(-(risk_score - 0.60) * 4.5))
        pd_val = float(np.clip(pd_val, 0.01, 0.99))

        # Stochastic default flag
        default = 1 if np.random.rand() < pd_val else 0

        data.append({
            'Category':             cat,
            'Age':                  age,
            'Monthly_Income':       income,
            'Savings':              savings,
            'Investments':          investments,
            'CIBIL_Score':          cibil,
            'Missed_EMIs':          missed_emis,
            'Credit_History_Length': credit_hist_len,
            'Existing_EMIs':        existing_emis,
            'DTI_Ratio':            dti,
            'Employment_Length':    emp_length,
            'Asset_Value':          asset_value,
            'Loan_Amount':          loan_amount,
            'Loan_Tenure':          tenure_months,
            'Loan_Purpose':         loan_purpose,
            'Co_Applicant':         co_applicant,
            'Loan_To_Income_Ratio': loan_to_income,
            'Net_Worth':            net_worth,
            'Default':              default
        })

    df = pd.DataFrame(data)

    out_dir = os.path.join(BASE_DIR, 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'borrowers.csv')
    df.to_csv(out_path, index=False)
    print(f"Generated {num_records} borrower records → {out_path}")
    print(f"Default rate: {df['Default'].mean() * 100:.2f}%")
    return df


if __name__ == "__main__":
    generate_borrower_dataset()
