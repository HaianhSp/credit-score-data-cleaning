"""Tạo đặc trưng từ dữ liệu đã làm sạch (không dùng cột Credit_Score).

BASE, RATIO (tỷ số tài chính), LOAN (từng loại khoản vay), PB_SPLIT (tách Payment_Behaviour),
LAG (theo thời gian, chỉ nhìn tháng hiện tại và trước đó), CUST_AGG (thống kê cả 8 tháng, chỉ để so sánh).
"""
import pandas as pd
from credit_cleaning import split_loan_types

LOAN_TYPES = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan',
              'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']
LAG_COLS = ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Utilization_Ratio', 'Num_Credit_Inquiries']
AGG_COLS = ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
            'Changed_Credit_Limit', 'Monthly_Balance', 'Amount_invested_monthly']

BASE_NUM = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
            'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
            'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Months',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
BASE_CAT = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
RATIO = ['Debt_to_Income', 'EMI_to_Salary', 'Invest_to_Salary', 'Balance_to_Salary', 'Implied_Spend_Ratio',
         'Debt_per_Loan', 'Has_Loan']
LOAN = ['Loan_' + t.replace(' ', '_').replace('-', '_') for t in LOAN_TYPES]
PB_SPLIT = ['Spend_Level', 'Payment_Size']
LAG = [f'{c}_{suffix}' for c in LAG_COLS for suffix in ('prev', 'delta', 'expmean')]
CUST_AGG = [f'{c}_cust_{suffix}' for c in AGG_COLS for suffix in ('mean', 'std')] + ['MinPay_No_share']

FEATURE_GROUPS = {'BASE': BASE_NUM + BASE_CAT, 'RATIO': RATIO, 'LOAN': LOAN, 'PB_SPLIT': PB_SPLIT,
                  'LAG': LAG, 'CUST_AGG': CUST_AGG}
CATEGORICAL = set(BASE_CAT + PB_SPLIT)


def add_ratio_features(d):
    d = d.copy()
    salary = d['Monthly_Inhand_Salary'].clip(lower=1)
    d['Debt_to_Income'] = d['Outstanding_Debt'] / d['Annual_Income']
    d['EMI_to_Salary'] = d['Total_EMI_per_month'] / salary
    d['Invest_to_Salary'] = d['Amount_invested_monthly'] / salary
    d['Balance_to_Salary'] = d['Monthly_Balance'] / salary
    d['Implied_Spend_Ratio'] = (salary - d['Total_EMI_per_month'] - d['Amount_invested_monthly']
                                - d['Monthly_Balance']) / salary
    d['Debt_per_Loan'] = d['Outstanding_Debt'] / d['Num_of_Loan'].clip(lower=1)
    d['Has_Loan'] = (d['Num_of_Loan'] > 0).astype(int)
    return d


def add_loan_flags(d):
    d = d.copy()
    types_of = {u: set(split_loan_types(u)) for u in d['Type_of_Loan'].unique()}
    for loan_type, col in zip(LOAN_TYPES, LOAN):
        d[col] = d['Type_of_Loan'].map(lambda u: int(loan_type in types_of[u]))
    return d


# 'Low_spent_Small_value_payments' -> Spend_Level = Low, Payment_Size = Small
def add_payment_split(d):
    d = d.copy()
    d['Spend_Level'] = d['Payment_Behaviour'].str.extract(r'^(High|Low)_spent')[0].fillna('Unknown')
    d['Payment_Size'] = d['Payment_Behaviour'].str.extract(r'_(Small|Medium|Large)_value')[0].fillna('Unknown')
    return d


# tháng đầu chưa có tháng trước nên lấy chính giá trị của nó
def add_lag_features(d):
    d = d.copy()
    by_cust = d.groupby('Customer_ID')
    for col in LAG_COLS:
        d[f'{col}_prev'] = by_cust[col].shift(1).fillna(d[col])
        d[f'{col}_delta'] = d[col] - d[f'{col}_prev']
        d[f'{col}_expmean'] = by_cust[col].cumsum() / d['Month_Num']
    return d


# dùng cả tháng sau nên mô hình chính không dùng
def add_customer_features(d):
    d = d.copy()
    by_cust = d.groupby('Customer_ID')
    for col in AGG_COLS:
        d[f'{col}_cust_mean'] = by_cust[col].transform('mean')
        d[f'{col}_cust_std'] = by_cust[col].transform('std')
    d['MinPay_No_share'] = (d['Payment_of_Min_Amount'] == 'No').groupby(d['Customer_ID']).transform('mean')
    return d


def build_features(clean_df):
    d = add_ratio_features(clean_df)
    d = add_loan_flags(d)
    d = add_payment_split(d)
    d = add_lag_features(d)
    d = add_customer_features(d)
    return d


def make_matrix(d, groups=('BASE',)):
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    X = d[cols].copy()
    categorical = [c for c in cols if c in CATEGORICAL]
    X = pd.get_dummies(X, columns=categorical, dtype=float)
    return X.astype(float)
