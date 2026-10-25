"""Làm sạch train.csv, hàm clean() chạy cả quy trình."""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
TARGET = 'Credit_Score'

NUMERIC_TEXT_COLS = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                     'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']

PLACEHOLDERS = {
    'SSN': '#F%$D@*&8',
    'Occupation': '_______',
    'Credit_Mix': '_',
    'Payment_Behaviour': '!@9#%8',
    'Changed_Credit_Limit': '_',
    'Amount_invested_monthly': '__10000__',
    'Monthly_Balance': '__-333333333333333333333333333__',
}

DOMAIN = {
    'Age': (14, 56),
    'Num_Bank_Accounts': (0, 10),
    'Num_Credit_Card': (1, 10),
    'Interest_Rate': (1, 34),
    'Num_of_Loan': (0, 9),
    'Num_of_Delayed_Payment': (0, 28),
    'Num_Credit_Inquiries': (0, 17),
}
EMI_MAX = 1800        # EMI thật lớn nhất là 1779
INCOME_REL_TOL = 0.5  # thu nhập lệch quá 50% so với trung vị của khách thì bỏ

REPAIR_PLAN = {
    'cust_mode':   ['SSN', 'Occupation', 'Name', 'Credit_Mix'],
    'cust_median': ['Annual_Income', 'Interest_Rate', 'Outstanding_Debt',
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit'],
    'nearest':     ['Num_Bank_Accounts', 'Num_Credit_Card', 'Age', 'Total_EMI_per_month',
                    'Monthly_Inhand_Salary', 'Num_Credit_Inquiries'],
    'model_ratio': ['Amount_invested_monthly', 'Monthly_Balance'],
}

MODEL_IMPUTE_FEATURES = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                         'Interest_Rate', 'Total_EMI_per_month', 'Outstanding_Debt', 'Age',
                         'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                         'Delay_from_due_date', 'Credit_Utilization_Ratio', 'Amount_invested_monthly',
                         'Monthly_Balance', 'Payment_Behaviour_Code', 'Month_Num']

INT_COLS = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
            'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_History_Months', 'Month_Num']

FINAL_COLUMN_ORDER = [
    'ID', 'Customer_ID', 'Month', 'Month_Num', 'Name', 'Age', 'SSN', 'Occupation', 'Annual_Income',
    'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
    'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
    'Credit_History_Months', 'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly',
    'Payment_Behaviour', 'Monthly_Balance', 'Credit_Score']


def add_month_num(d):
    d = d.copy()
    d['Month_Num'] = d['Month'].map({m: i + 1 for i, m in enumerate(MONTHS)}).astype(int)
    return d


# Bước 1: giá trị giả thành NaN
def mark_placeholders(d, log=None):
    d = d.copy()
    for col, token in PLACEHOLDERS.items():
        is_fake = d[col] == token
        if log is not None:
            log.setdefault('placeholder', {})[col] = int(is_fake.sum())
        d.loc[is_fake, col] = np.nan
    return d


# Bước 2: bỏ dấu _ rồi ép số. Phải chạy sau bước 1, không thì "__10000__" thành số 10000
def strip_numeric_text(d):
    d = d.copy()
    for col in NUMERIC_TEXT_COLS:
        text = d[col].astype('string').str.replace(r'^_+|_+$', '', regex=True)
        d[col] = pd.to_numeric(text, errors='coerce').astype('float64')
    return d


# Bước 3: giá trị vô lý thành NaN
def mark_invalid(d, log=None):
    d = d.copy()
    log = {} if log is None else log
    invalid_count = log.setdefault('invalid', {})

    for col, (low, high) in DOMAIN.items():
        out_of_range = (d[col] < low) | (d[col] > high)
        invalid_count[col] = int(out_of_range.sum())
        d.loc[out_of_range, col] = np.nan

    emi = d['Total_EMI_per_month']
    bad_emi = (emi > EMI_MAX) | ((emi > 0) & (emi == emi.round(0)))  # EMI thật có số lẻ, số tròn là rác
    invalid_count['Total_EMI_per_month'] = int(bad_emi.sum())
    d.loc[bad_emi, 'Total_EMI_per_month'] = np.nan

    median_income = d.groupby('Customer_ID')['Annual_Income'].transform('median')
    bad_income = (d['Annual_Income'] / median_income - 1).abs() > INCOME_REL_TOL
    invalid_count['Annual_Income'] = int(bad_income.sum())
    d.loc[bad_income, 'Annual_Income'] = np.nan
    return d


# Bước 4: điền ô thiếu
def nearest_fill(d, col):
    t = pd.DataFrame({'cust': d['Customer_ID'], 'month': d['Month_Num'], 'value': d[col]})
    t['valid_month'] = t['month'].where(t['value'].notna())
    by_cust = t.groupby('cust')
    prev_value, prev_month = by_cust['value'].ffill(), by_cust['valid_month'].ffill()
    next_value, next_month = by_cust['value'].bfill(), by_cust['valid_month'].bfill()
    gap_prev, gap_next = t['month'] - prev_month, next_month - t['month']
    take_prev = gap_prev.notna() & (gap_next.isna() | (gap_prev <= gap_next))
    filled = pd.Series(np.where(take_prev, prev_value, next_value), index=d.index)
    return filled.where(t['value'].isna(), t['value'])


def most_common(s):
    modes = s.dropna().mode()
    return modes.iloc[0] if len(modes) else np.nan


def payment_behaviour_code(d):
    categories = sorted(d['Payment_Behaviour'].dropna().unique())
    return d['Payment_Behaviour'].map({c: i for i, c in enumerate(categories)}).fillna(-1)


def salary_for_scaling(d):
    return fill_numeric(d, 'Monthly_Inhand_Salary', 'nearest').clip(lower=1)


# ratio=True: dự đoán tỷ lệ so với lương rồi nhân lại
def model_fill(d, col, features=MODEL_IMPUTE_FEATURES, seed=0, ratio=False):
    X = d.assign(Payment_Behaviour_Code=payment_behaviour_code(d))[[f for f in features if f != col]]
    y = d[col]
    if ratio:
        salary = salary_for_scaling(d)
        for other in ('Amount_invested_monthly', 'Monthly_Balance', 'Total_EMI_per_month'):
            if other != col:
                X[other + '_to_salary'] = d[other] / salary
        X['Debt_to_Income'] = d['Outstanding_Debt'] / d['Annual_Income']
        y = y / salary
    known = y.notna()
    model = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(X[known], y[known])
    pred = pd.Series(np.clip(model.predict(X), 0, None), index=d.index)
    if ratio:
        pred = pred * salary
    return d[col].where(d[col].notna(), pred)


def fill_numeric(d, col, strategy):
    s = d[col]
    if strategy == 'global_median':
        filled = s.fillna(s.median())
    elif strategy == 'cust_median':
        filled = s.fillna(d.groupby('Customer_ID')[col].transform('median'))
    elif strategy == 'nearest':
        filled = nearest_fill(d, col)
    elif strategy == 'interpolate':
        filled = d.groupby('Customer_ID')[col].transform(lambda x: x.interpolate(limit_direction='both'))
    elif strategy == 'model':
        filled = model_fill(d, col)
    elif strategy == 'model_ratio':
        filled = model_fill(d, col, ratio=True)
    else:
        raise ValueError(strategy)
    return filled.fillna(s.median())


def fill_categorical(d, col, strategy):
    s = d[col]
    if strategy == 'global_mode':
        filled = s.fillna(most_common(s))
    elif strategy == 'cust_mode':
        filled = s.fillna(d.groupby('Customer_ID')[col].transform(most_common))
    elif strategy == 'nearest':
        codes = s.astype('category')
        near = nearest_fill(d.assign(**{col: codes.cat.codes.replace(-1, np.nan)}), col)
        filled = s.fillna(near.map(lambda x: codes.cat.categories[int(x)] if pd.notna(x) else np.nan))
    else:
        raise ValueError(strategy)
    return filled.fillna(most_common(s))


# '22 Years and 9 Months' -> 273
def parse_history_months(x):
    m = re.match(r'(\d+) Years and (\d+) Months', str(x))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def split_loan_types(x):
    parts = re.split(r',\s*', str(x).replace(' and ', ' '))
    return [p.strip() for p in parts if p.strip()]


def count_loan_types(x):
    if pd.isna(x) or x == 'No Loan':
        return 0
    return len(split_loan_types(x))


def fill_columns(d, cols, strategy, log):
    for col in cols:
        log[col] = int(d[col].isna().sum())
        if strategy == 'cust_mode':
            d[col] = fill_categorical(d, col, strategy)
        else:
            d[col] = fill_numeric(d, col, strategy)


def rebuild_loan_count(d, log):
    log['Num_of_Loan'] = int(d['Num_of_Loan'].isna().sum())
    d['Num_of_Loan'] = d['Type_of_Loan'].map(count_loan_types)


def rebuild_credit_history(d, log):
    months = d['Credit_History_Age'].map(parse_history_months)
    offset = (months - d['Month_Num']).groupby(d['Customer_ID']).transform('median')
    log['Credit_History_Age'] = int(months.isna().sum())
    d['Credit_History_Months'] = months.fillna(offset + d['Month_Num']).fillna(months.median())
    d.drop(columns='Credit_History_Age', inplace=True)


def repair(d, plan=REPAIR_PLAN, log=None):
    d = d.copy()
    log = {} if log is None else log
    repaired = log.setdefault('repaired', {})

    # không cách nào đoán tốt hơn ngẫu nhiên nên giữ thành nhóm Unknown
    repaired['Payment_Behaviour'] = int(d['Payment_Behaviour'].isna().sum())
    d['Payment_Behaviour'] = d['Payment_Behaviour'].fillna('Unknown')

    # trống nghĩa là khách không vay
    repaired['Type_of_Loan'] = int(d['Type_of_Loan'].isna().sum())
    d['Type_of_Loan'] = d['Type_of_Loan'].fillna('No Loan')

    fill_columns(d, plan['cust_mode'], 'cust_mode', repaired)
    fill_columns(d, plan['cust_median'], 'cust_median', repaired)
    fill_columns(d, plan['nearest'], 'nearest', repaired)

    rebuild_loan_count(d, repaired)
    rebuild_credit_history(d, repaired)

    # hai cột tiền điền cuối vì cần các cột khác đã đủ
    fill_columns(d, plan['model_ratio'], 'model_ratio', repaired)
    return d


def finalize(d):
    d = d.copy()
    for col in INT_COLS:
        d[col] = d[col].round().astype(int)
    return d[[c for c in FINAL_COLUMN_ORDER if c in d.columns]]


def clean(df_raw, return_log=False):
    log = {}
    d = add_month_num(df_raw)
    d = mark_placeholders(d, log)
    d = strip_numeric_text(d)
    d = mark_invalid(d, log)
    d = repair(d, log=log)
    d = finalize(d)
    return (d, log) if return_log else d


def validate_clean(d):
    by_cust = d.groupby('Customer_ID')
    return {
        'Không còn ô thiếu': int(d.isna().sum().sum()) == 0,
        'Không còn placeholder': all((d[c].astype(str) != t).all() for c, t in PLACEHOLDERS.items() if c in d),
        'Mọi cột đếm nằm trong miền hợp lệ': all(d[c].between(lo, hi).all() for c, (lo, hi) in DOMAIN.items()),
        'EMI trong [0, 1800]': bool(d['Total_EMI_per_month'].between(0, EMI_MAX).all()),
        'Mỗi khách đúng 1 SSN / nghề / tên / Credit_Mix': all(by_cust[c].nunique().eq(1).all()
                                                           for c in ['SSN', 'Occupation', 'Name', 'Credit_Mix']),
        'Mỗi khách 1 mức Annual_Income / Interest_Rate / Outstanding_Debt': all(
            by_cust[c].nunique().eq(1).all() for c in ['Annual_Income', 'Interest_Rate', 'Outstanding_Debt']),
        'Num_of_Loan = số loại vay': bool((d['Type_of_Loan'].map(count_loan_types) == d['Num_of_Loan']).all()),
        'Khách "No Loan" ⇔ Num_of_Loan = 0': bool(((d['Type_of_Loan'] == 'No Loan') == (d['Num_of_Loan'] == 0)).all()),
        'Tuổi trong 1 khách chênh tối đa 1': bool((by_cust['Age'].max() - by_cust['Age'].min()).le(1).all()),
        'Credit_History tăng đúng +1/tháng': bool((d['Credit_History_Months'] - d['Month_Num'])
                                                   .groupby(d['Customer_ID']).nunique().eq(1).all()),
        'Monthly_Balance & Amount_invested ≥ 0': bool((d[['Monthly_Balance', 'Amount_invested_monthly']] >= 0).all().all()),
        '8 dòng / khách, đúng thứ tự tháng': bool(by_cust['Month_Num'].apply(lambda s: list(s) == list(range(1, 9))).all()),
    }
