"""
credit_features.py
==================
Tạo đặc trưng (feature engineering) trên dữ liệu ĐÃ LÀM SẠCH (output của credit_cleaning.clean).
Dùng lại ở notebook 3 (phân tích đặc trưng) và notebook 4 (mô hình).

Nhóm đặc trưng:
    BASE_NUM / BASE_CAT  – các cột gốc đã sạch
    RATIO                – tỷ số tài chính theo từng dòng (nợ/thu nhập, EMI/lương, ...)
    LOAN                 – cờ 0/1 cho từng loại khoản vay
    PB_SPLIT             – tách Payment_Behaviour thành mức chi tiêu × độ lớn thanh toán
    LAG                  – đặc trưng "nhân quả" chỉ dùng tháng hiện tại và các tháng TRƯỚC
    CUST_AGG             – thống kê trên cả 8 tháng của khách ("nhìn lại"; dùng thông tin các tháng sau)

Không dùng nhãn Credit_Score để tạo đặc trưng.
"""
import re
import numpy as np
import pandas as pd

LOAN_TYPES = ['Auto Loan', 'Credit-Builder Loan', 'Debt Consolidation Loan', 'Home Equity Loan', 'Mortgage Loan',
              'Not Specified', 'Payday Loan', 'Personal Loan', 'Student Loan']
AGG_COLS = ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
            'Changed_Credit_Limit', 'Monthly_Balance', 'Amount_invested_monthly']
LAG_COLS = ['Delay_from_due_date', 'Num_of_Delayed_Payment', 'Credit_Utilization_Ratio', 'Num_Credit_Inquiries']

BASE_NUM = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
            'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
            'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Months',
            'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
BASE_CAT = ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']
RATIO = ['Debt_to_Income', 'EMI_to_Salary', 'Invest_to_Salary', 'Balance_to_Salary', 'Implied_Spend_Ratio',
         'Debt_per_Loan', 'Has_Loan']
LOAN = ['Loan_' + t.replace(' ', '_').replace('-', '_') for t in LOAN_TYPES]
PB_SPLIT = ['Spend_Level', 'Payment_Size']
LAG = [f'{c}_{s}' for c in LAG_COLS for s in ('prev', 'delta', 'expmean')]
CUST_AGG = [f'{c}_cust_{s}' for c in AGG_COLS for s in ('mean', 'std')] + ['MinPay_No_share']

FEATURE_GROUPS = {'BASE': BASE_NUM + BASE_CAT, 'RATIO': RATIO, 'LOAN': LOAN, 'PB_SPLIT': PB_SPLIT,
                  'LAG': LAG, 'CUST_AGG': CUST_AGG}
CATEGORICAL = set(BASE_CAT + PB_SPLIT)


def add_ratio_features(d):
    """Tỷ số tài chính theo từng dòng. Mẫu số được chặn dưới để tránh chia cho 0."""
    d = d.copy()
    salary = d['Monthly_Inhand_Salary'].clip(lower=1)
    d['Debt_to_Income'] = d['Outstanding_Debt'] / d['Annual_Income']
    d['EMI_to_Salary'] = d['Total_EMI_per_month'] / salary
    d['Invest_to_Salary'] = d['Amount_invested_monthly'] / salary
    d['Balance_to_Salary'] = d['Monthly_Balance'] / salary
    d['Implied_Spend_Ratio'] = (salary - d['Total_EMI_per_month'] - d['Amount_invested_monthly'] - d['Monthly_Balance']) / salary
    d['Debt_per_Loan'] = d['Outstanding_Debt'] / d['Num_of_Loan'].clip(lower=1)
    d['Has_Loan'] = (d['Num_of_Loan'] > 0).astype(int)
    return d


def add_loan_flags(d):
    """Mỗi loại khoản vay -> cột 0/1 (tách chuỗi 'Auto Loan, Credit-Builder Loan, and Home Equity Loan')."""
    d = d.copy()
    uniq = d['Type_of_Loan'].unique()
    parsed = {u: {p.strip() for p in re.split(r',\s*', str(u).replace(' and ', ' '))} for u in uniq}
    for t, name in zip(LOAN_TYPES, LOAN):
        d[name] = d['Type_of_Loan'].map(lambda u: int(t in parsed[u]))
    return d


def add_payment_split(d):
    """Payment_Behaviour = '<Mức chi tiêu>_spent_<Độ lớn>_value_payments' -> hai cột riêng (Unknown giữ nguyên)."""
    d = d.copy()
    d['Spend_Level'] = d['Payment_Behaviour'].str.extract(r'^(High|Low)_spent')[0].fillna('Unknown')
    d['Payment_Size'] = d['Payment_Behaviour'].str.extract(r'_(Small|Medium|Large)_value')[0].fillna('Unknown')
    return d


def add_lag_features(d):
    """Đặc trưng nhân quả: giá trị tháng trước, thay đổi so với tháng trước, trung bình lũy kế tới tháng hiện tại.
    Tháng đầu tiên (không có tháng trước) dùng chính giá trị hiện tại."""
    d = d.copy()
    g = d.groupby('Customer_ID')
    for c in LAG_COLS:
        d[f'{c}_prev'] = g[c].shift(1).fillna(d[c])
        d[f'{c}_delta'] = d[c] - d[f'{c}_prev']
        d[f'{c}_expmean'] = g[c].cumsum() / d['Month_Num']
    return d


def add_customer_features(d):
    """Thống kê trên cả 8 tháng của khách (trung bình, độ lệch chuẩn, tỷ lệ tháng không trả mức tối thiểu).
    Lưu ý: dùng thông tin của các tháng SAU tháng hiện tại -> chỉ hợp lệ khi đánh giá 'nhìn lại' trọn 8 tháng."""
    d = d.copy()
    g = d.groupby('Customer_ID')
    for c in AGG_COLS:
        d[f'{c}_cust_mean'] = g[c].transform('mean')
        d[f'{c}_cust_std'] = g[c].transform('std')
    d['MinPay_No_share'] = (d['Payment_of_Min_Amount'] == 'No').groupby(d['Customer_ID']).transform('mean')
    return d


def build_features(clean_df):
    """clean_df (output của credit_cleaning.clean) -> bảng có thêm mọi nhóm đặc trưng."""
    d = add_ratio_features(clean_df)
    d = add_loan_flags(d)
    d = add_payment_split(d)
    d = add_lag_features(d)
    d = add_customer_features(d)
    return d


def make_matrix(d, groups=('BASE',)):
    """Ma trận đặc trưng số (one-hot cho cột phân loại) từ các nhóm đã chọn. Cột đầu ra ổn định giữa các lần gọi."""
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    X = d[cols].copy()
    cat = [c for c in cols if c in CATEGORICAL]
    X = pd.get_dummies(X, columns=cat, dtype=float)
    return X.astype(float)
