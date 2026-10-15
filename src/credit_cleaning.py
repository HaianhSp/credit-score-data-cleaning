"""
credit_cleaning.py
==================
Pipeline làm sạch cho bộ dữ liệu Credit Score Classification (train.csv).

Mọi quyết định trong file này đều có bằng chứng ở notebook 1 (EDA) và được kiểm chứng ở notebook 2.
File này là "nguồn duy nhất" (single source of truth): notebook 2, 3, 4 đều dùng lại các hàm ở đây.

Thứ tự pipeline:
    1. mark_placeholders   – giá trị giả  -> NaN            (phải làm TRƯỚC khi bỏ dấu '_')
    2. strip_numeric_text  – '24_' -> 24, ép kiểu số
    3. mark_invalid        – giá trị vô lý (ngoài miền hợp lệ, chữ ký dữ liệu tiêm, lệch khách hàng) -> NaN
    4. repair              – sửa/điền theo từng cột bằng chiến lược đã kiểm chứng
    5. finalize            – ép kiểu, chuyển Credit_History_Age sang số tháng

Lưu ý: pipeline KHÔNG BAO GIỜ dùng cột nhãn `Credit_Score` (tránh rò rỉ).
"""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ----------------------------------------------------------------------------------------------
# Hằng số – rút ra từ EDA (notebook 1)
# ----------------------------------------------------------------------------------------------
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
ID_COLS = ['ID', 'Customer_ID', 'Name', 'SSN']
TARGET = 'Credit_Score'

# Các cột lẽ ra là số nhưng bị lưu thành chữ có ký tự '_' dính vào
NUMERIC_TEXT_COLS = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                     'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']

# Giá trị giả (placeholder) – nhận diện ở EDA mục 1.4
PLACEHOLDERS = {
    'SSN': '#F%$D@*&8',
    'Occupation': '_______',
    'Credit_Mix': '_',
    'Payment_Behaviour': '!@9#%8',
    'Changed_Credit_Limit': '_',
    'Amount_invested_monthly': '__10000__',
    'Monthly_Balance': '__-333333333333333333333333333__',
}

# Miền hợp lệ (đóng) = "thân" phân phối: đoạn giá trị liên tiếp mà mỗi giá trị xuất hiện >= 50 lần (cột đếm: cận dưới >= 0)
# (EDA mục 1.5: phía trên cận này mỗi giá trị chỉ xuất hiện vài lần => nhiễu tiêm vào, không phải phân phối thật)
DOMAIN = {
    'Age': (14, 56),
    'Num_Bank_Accounts': (0, 10),
    'Num_Credit_Card': (1, 10),
    'Interest_Rate': (1, 34),
    'Num_of_Loan': (0, 9),
    'Num_of_Delayed_Payment': (0, 28),
    'Num_Credit_Inquiries': (0, 17),
}
EMI_MAX = 1800  # EMI hợp lệ lớn nhất quan sát được là 1779; mọi số nguyên dương còn lại là nhiễu tiêm
INCOME_REL_TOL = 0.5  # Annual_Income lệch > 50% so với trung vị của chính khách hàng => nhiễu (thu nhập thật không đổi trong 8 tháng)

# Kế hoạch sửa lỗi theo từng cột – chọn bằng thí nghiệm "che 10% giá trị" ở notebook 2
REPAIR_PLAN = {
    'cust_mode':   ['SSN', 'Occupation', 'Name', 'Credit_Mix'],
    'cust_median': ['Annual_Income', 'Interest_Rate', 'Outstanding_Debt',
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit'],
    'nearest':     ['Num_Bank_Accounts', 'Num_Credit_Card', 'Age', 'Total_EMI_per_month',
                    'Monthly_Inhand_Salary', 'Num_Credit_Inquiries'],
    'model_ratio': ['Amount_invested_monthly', 'Monthly_Balance'],
}

# Các cột làm đặc trưng cho mô hình điền thiếu (không chứa nhãn)
MODEL_IMPUTE_FEATURES = ['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card',
                         'Interest_Rate', 'Total_EMI_per_month', 'Outstanding_Debt', 'Age',
                         'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
                         'Delay_from_due_date', 'Credit_Utilization_Ratio', 'Amount_invested_monthly',
                         'Monthly_Balance', 'Payment_Behaviour_Code', 'Month_Num']


# ----------------------------------------------------------------------------------------------
# Bước 1–2: kiểu dữ liệu và giá trị giả
# ----------------------------------------------------------------------------------------------
def add_month_num(d):
    """Thêm Month_Num (1..8). Dữ liệu được sắp theo (Customer_ID, Month) nên đây cũng là thứ tự thời gian."""
    d = d.copy()
    d['Month_Num'] = d['Month'].map({m: i + 1 for i, m in enumerate(MONTHS)}).astype(int)
    return d


def mark_placeholders(d, log=None):
    """Bước 1: đổi các giá trị giả sang NaN. Phải chạy TRƯỚC strip_numeric_text,
    vì '__10000__' sẽ trở thành số 10000 'trông như thật' nếu chỉ bỏ dấu '_'."""
    d = d.copy()
    for col, token in PLACEHOLDERS.items():
        bad = d[col] == token
        if log is not None:
            log.setdefault('placeholder', {})[col] = int(bad.sum())
        d.loc[bad, col] = np.nan
    return d


def strip_numeric_text(d):
    """Bước 2: bỏ dấu '_' ở đầu/cuối rồi ép kiểu số ('24_' -> 24, '14388.79_' -> 14388.79)."""
    d = d.copy()
    for col in NUMERIC_TEXT_COLS:
        txt = d[col].astype('string').str.replace(r'^_+|_+$', '', regex=True)
        d[col] = pd.to_numeric(txt, errors='coerce').astype('float64')   # float64 để NaN hoạt động bình thường
    return d


# ----------------------------------------------------------------------------------------------
# Bước 3: giá trị vô lý -> NaN
# ----------------------------------------------------------------------------------------------
def mark_invalid(d, log=None):
    """Bước 3: đánh dấu NaN cho giá trị không thể là dữ liệu thật.
    (a) ngoài miền hợp lệ DOMAIN;  (b) EMI: lớn bất thường hoặc là số nguyên dương
    (EMI thật là số thực dài ~15 chữ số lẻ, số nguyên dương chỉ có ở giá trị nhiễu);
    (c) Annual_Income lệch > 50% so với trung vị của chính khách hàng."""
    d = d.copy()
    log = {} if log is None else log
    inv = log.setdefault('invalid', {})
    for col, (lo, hi) in DOMAIN.items():
        bad = (d[col] < lo) | (d[col] > hi)
        inv[col] = int(bad.sum())
        d.loc[bad, col] = np.nan
    emi = d['Total_EMI_per_month']
    bad = (emi > EMI_MAX) | ((emi > 0) & (emi == emi.round(0)))
    inv['Total_EMI_per_month'] = int(bad.sum())
    d.loc[bad, 'Total_EMI_per_month'] = np.nan
    rel_dev = (d['Annual_Income'] / d.groupby('Customer_ID')['Annual_Income'].transform('median') - 1).abs()
    bad = rel_dev > INCOME_REL_TOL
    inv['Annual_Income'] = int(bad.sum())
    d.loc[bad, 'Annual_Income'] = np.nan
    return d


# ----------------------------------------------------------------------------------------------
# Bước 4: sửa / điền thiếu – các chiến lược dùng chung cho cả thí nghiệm và pipeline
# ----------------------------------------------------------------------------------------------
def nearest_fill(d, col):
    """Điền ô thiếu bằng giá trị ở tháng GẦN NHẤT (cùng khách hàng) còn hợp lệ; hòa thì lấy tháng trước.
    Phù hợp cột có 'bước nhảy' (số tài khoản, lương, EMI): không kéo giá trị của giai đoạn này sang giai đoạn khác."""
    t = pd.DataFrame({'c': d['Customer_ID'], 'm': d['Month_Num'], 'v': d[col]})
    t['vm'] = t['m'].where(t['v'].notna())
    g = t.groupby('c')
    prev_v, prev_m = g['v'].ffill(), g['vm'].ffill()
    next_v, next_m = g['v'].bfill(), g['vm'].bfill()
    d_prev, d_next = t['m'] - prev_m, next_m - t['m']
    use_prev = d_prev.notna() & (d_next.isna() | (d_prev <= d_next))
    return pd.Series(np.where(use_prev, prev_v, next_v), index=d.index).where(t['v'].isna(), t['v'])


def _cust_mode(s):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else np.nan


def payment_behaviour_code(d):
    """Mã hóa Payment_Behaviour thành số (NaN/Unknown có mã riêng) để dùng làm đặc trưng cho mô hình điền thiếu."""
    cats = sorted(d['Payment_Behaviour'].dropna().unique())
    return d['Payment_Behaviour'].map({c: i for i, c in enumerate(cats)}).fillna(-1)


def model_fill(d, col, features=MODEL_IMPUTE_FEATURES, seed=0, ratio=False):
    """Điền cột `col` bằng HistGradientBoostingRegressor học từ các cột còn lại của CÙNG dòng
    (Amount_invested_monthly và Monthly_Balance phụ thuộc mạnh vào lương và hành vi chi tiêu).
    ratio=True: dự đoán *tỷ lệ so với lương* (col / Monthly_Inhand_Salary) rồi nhân lại. Số tiền đầu tư / số dư tỷ lệ
    với thu nhập nên tỷ lệ ổn định hơn giữa các mức lương; trong thí nghiệm che giá trị (notebook 2, mục 4.2) cách này
    cho sai số thấp hơn dự đoán trực tiếp giá trị tuyệt đối, ở mọi mức lương."""
    X = d.assign(Payment_Behaviour_Code=payment_behaviour_code(d))[[f for f in features if f != col]]
    y = d[col]
    if ratio:
        scale = fill_numeric(d, 'Monthly_Inhand_Salary', 'nearest').clip(lower=1)
        for c in ('Amount_invested_monthly', 'Monthly_Balance', 'Total_EMI_per_month'):
            if c != col:
                X[c + '_to_salary'] = d[c] / scale
        X['Debt_to_Income'] = d['Outstanding_Debt'] / d['Annual_Income']
        y = y / scale
    known = y.notna()
    model = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(X[known], y[known])
    pred = pd.Series(np.clip(model.predict(X), 0, None), index=d.index)
    if ratio:
        pred = pred * scale
    return d[col].where(d[col].notna(), pred)


def fill_numeric(d, col, strategy):
    """Trả về cột `col` đã điền theo `strategy` ∈ {global_median, cust_median, nearest, interpolate, model, model_ratio}."""
    s = d[col]
    if strategy == 'global_median':
        out = s.fillna(s.median())
    elif strategy == 'cust_median':
        out = s.fillna(d.groupby('Customer_ID')[col].transform('median'))
    elif strategy == 'nearest':
        out = nearest_fill(d, col)
    elif strategy == 'interpolate':
        out = d.groupby('Customer_ID')[col].transform(lambda x: x.interpolate(limit_direction='both'))
    elif strategy == 'model':
        out = model_fill(d, col)
    elif strategy == 'model_ratio':
        out = model_fill(d, col, ratio=True)
    else:
        raise ValueError(strategy)
    return out.fillna(s.median())          # dự phòng: khách không còn giá trị hợp lệ nào


def fill_categorical(d, col, strategy):
    """Trả về cột phân loại đã điền theo `strategy` ∈ {global_mode, cust_mode, nearest}."""
    s = d[col]
    if strategy == 'global_mode':
        out = s.fillna(_cust_mode(s))
    elif strategy == 'cust_mode':
        out = s.fillna(d.groupby('Customer_ID')[col].transform(_cust_mode))
    elif strategy == 'nearest':
        codes = s.astype('category')
        near = nearest_fill(d.assign(**{col: codes.cat.codes.replace(-1, np.nan)}), col)
        out = s.fillna(near.map(lambda x: codes.cat.categories[int(x)] if pd.notna(x) else np.nan))
    else:
        raise ValueError(strategy)
    return out.fillna(_cust_mode(s))


def parse_history_months(x):
    """'22 Years and 9 Months' -> 273 (tháng)."""
    m = re.match(r'(\d+) Years and (\d+) Months', str(x))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def count_loan_types(x):
    """'Auto Loan, Credit-Builder Loan, and Home Equity Loan' -> 3;  NaN/'No Loan' -> 0."""
    if pd.isna(x) or x == 'No Loan':
        return 0
    return len([p for p in re.split(r',\s*', str(x).replace(' and ', ' ')) if p.strip()])


def repair(d, plan=REPAIR_PLAN, log=None):
    """Bước 4: sửa lỗi theo kế hoạch REPAIR_PLAN + các luật xác định (deterministic)."""
    d = d.copy()
    log = {} if log is None else log
    rep = log.setdefault('repaired', {})
    # --- Payment_Behaviour: ngẫu nhiên theo tháng, không chiến lược nào đoán tốt hơn ngẫu nhiên (notebook 2)
    #     => giữ thông tin "không rõ" bằng một nhóm riêng thay vì bịa giá trị
    rep['Payment_Behaviour'] = int(d['Payment_Behaviour'].isna().sum())
    d['Payment_Behaviour'] = d['Payment_Behaviour'].fillna('Unknown')
    # --- Type_of_Loan: ô trống = khách KHÔNG có khoản vay nào (thiếu đủ 8 tháng và Num_of_Loan = 0, EDA 1.3)
    #     => đây là một giá trị thật ('No Loan'), tuyệt đối không điền bằng giá trị phổ biến
    rep['Type_of_Loan'] = int(d['Type_of_Loan'].isna().sum())
    d['Type_of_Loan'] = d['Type_of_Loan'].fillna('No Loan')
    # --- Cột phân loại hằng số theo khách
    for col in plan['cust_mode']:
        n = int(d[col].isna().sum())
        d[col] = fill_categorical(d, col, 'cust_mode')
        rep[col] = n
    # --- Số liệu: median theo khách / tháng gần nhất
    for strat in ('cust_median', 'nearest'):
        for col in plan[strat]:
            n = int(d[col].isna().sum())
            d[col] = fill_numeric(d, col, strat)
            rep[col] = n
    # --- Luật xác định: Num_of_Loan = số loại vay liệt kê trong Type_of_Loan (khớp 100% ở dữ liệu hợp lệ)
    rep['Num_of_Loan'] = int(d['Num_of_Loan'].isna().sum())
    d['Num_of_Loan'] = d['Type_of_Loan'].map(count_loan_types)
    # --- Credit_History_Age: tăng đúng +1 tháng mỗi tháng => suy từ (giá trị − thứ tự tháng) của các tháng còn lại
    hist = d['Credit_History_Age'].map(parse_history_months)
    offset = (hist - d['Month_Num']).groupby(d['Customer_ID']).transform('median')
    rep['Credit_History_Age'] = int(hist.isna().sum())
    d['Credit_History_Months'] = hist.fillna(offset + d['Month_Num']).fillna(hist.median())
    d = d.drop(columns='Credit_History_Age')
    # --- Mô hình học máy (dự đoán tỷ lệ so với lương) cho hai cột phụ thuộc mạnh vào lương/hành vi chi tiêu
    for col in plan['model_ratio']:
        n = int(d[col].isna().sum())
        d[col] = fill_numeric(d, col, 'model_ratio')
        rep[col] = n
    return d


# ----------------------------------------------------------------------------------------------
# Bước 5 + pipeline tổng
# ----------------------------------------------------------------------------------------------
INT_COLS = ['Age', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
            'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_History_Months', 'Month_Num']


def finalize(d):
    """Bước 5: ép kiểu cuối cùng và sắp xếp cột."""
    d = d.copy()
    for c in INT_COLS:
        d[c] = d[c].round().astype(int)
    order = ['ID', 'Customer_ID', 'Month', 'Month_Num', 'Name', 'Age', 'SSN', 'Occupation', 'Annual_Income',
             'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan',
             'Type_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
             'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 'Credit_Utilization_Ratio',
             'Credit_History_Months', 'Payment_of_Min_Amount', 'Total_EMI_per_month', 'Amount_invested_monthly',
             'Payment_Behaviour', 'Monthly_Balance', 'Credit_Score']
    return d[[c for c in order if c in d.columns]]


def clean(df_raw, return_log=False):
    """Pipeline đầy đủ: df_raw (train.csv) -> dữ liệu sạch."""
    log = {}
    d = add_month_num(df_raw)
    d = mark_placeholders(d, log)
    d = strip_numeric_text(d)
    d = mark_invalid(d, log)
    d = repair(d, log=log)
    d = finalize(d)
    return (d, log) if return_log else d


def validate_clean(d):
    """Các kiểm tra ràng buộc sau làm sạch. Trả về dict {tên kiểm tra: True/False}."""
    g = d.groupby('Customer_ID')
    checks = {
        'Không còn ô thiếu': int(d.isna().sum().sum()) == 0,
        'Không còn placeholder': all((d[c].astype(str) != t).all() for c, t in PLACEHOLDERS.items() if c in d),
        'Mọi cột đếm nằm trong miền hợp lệ': all(d[c].between(lo, hi).all() for c, (lo, hi) in DOMAIN.items()),
        'EMI trong [0, 1800]': bool(d['Total_EMI_per_month'].between(0, EMI_MAX).all()),
        'Mỗi khách đúng 1 SSN / nghề / tên / Credit_Mix': all(g[c].nunique().eq(1).all()
                                                          for c in ['SSN', 'Occupation', 'Name', 'Credit_Mix']),
        'Mỗi khách 1 mức Annual_Income / Interest_Rate / Outstanding_Debt': all(g[c].nunique().eq(1).all()
                                                          for c in ['Annual_Income', 'Interest_Rate', 'Outstanding_Debt']),
        'Num_of_Loan = số loại vay': bool((d['Type_of_Loan'].map(count_loan_types) == d['Num_of_Loan']).all()),
        'Khách "No Loan" ⇔ Num_of_Loan = 0': bool(((d['Type_of_Loan'] == 'No Loan') == (d['Num_of_Loan'] == 0)).all()),
        'Tuổi trong 1 khách chênh tối đa 1': bool((g['Age'].max() - g['Age'].min()).le(1).all()),
        'Credit_History tăng đúng +1/tháng': bool((d['Credit_History_Months'] - d['Month_Num'])
                                                    .groupby(d['Customer_ID']).nunique().eq(1).all()),
        'Monthly_Balance & Amount_invested ≥ 0': bool((d[['Monthly_Balance', 'Amount_invested_monthly']] >= 0).all().all()),
        '8 dòng / khách, đúng thứ tự tháng': bool(g['Month_Num'].apply(lambda s: list(s) == list(range(1, 9))).all()),
    }
    return checks
