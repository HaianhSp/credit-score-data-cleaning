"""
Làm sạch dữ liệu Credit Score Classification (train.csv).

File này chứa toàn bộ các hàm làm sạch. Notebook 2, 3, 4 đều import lại từ đây,
nên code chỉ nằm ở một chỗ. Các hàm được xếp theo đúng thứ tự pipeline:

    (chuẩn bị) add_month_num   thêm số thứ tự tháng (1 đến 8)
    1. mark_placeholders       giá trị giả (vd "_______", "#F%$D@*&8") -> NaN
    2. strip_numeric_text      bỏ dấu "_" thừa rồi ép kiểu số ("24_" -> 24)
    3. mark_invalid            giá trị vô lý (ngoài miền hợp lệ, lệch quá xa) -> NaN
    4. repair                  điền lại các ô NaN, mỗi cột một cách đã chọn bằng thí nghiệm ở notebook 2
    (hoàn thiện) finalize      ép kiểu số nguyên, sắp lại thứ tự cột

Bước 2 phải chạy sau bước 1, nếu không "__10000__" bị bỏ dấu "_" thành số 10000 trông như thật.

Cột nhãn Credit_Score không được dùng ở bất kỳ bước nào, để tránh rò rỉ nhãn khi huấn luyện.
"""
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------------------
# Hằng số (mỗi hằng số có bằng chứng ở notebook 1)
# ---------------------------------------------------------------------------
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
TARGET = 'Credit_Score'

# Các cột lẽ ra là số nhưng bị lưu thành chữ, có dấu "_" dính ở đầu hoặc cuối
NUMERIC_TEXT_COLS = ['Age', 'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                     'Outstanding_Debt', 'Amount_invested_monthly', 'Monthly_Balance']

# Giá trị giả tìm được ở notebook 1 (mục 1.4)
PLACEHOLDERS = {
    'SSN': '#F%$D@*&8',
    'Occupation': '_______',
    'Credit_Mix': '_',
    'Payment_Behaviour': '!@9#%8',
    'Changed_Credit_Limit': '_',
    'Amount_invested_monthly': '__10000__',
    'Monthly_Balance': '__-333333333333333333333333333__',
}

# Miền hợp lệ của các cột đếm. Cách tìm: lấy đoạn giá trị liền nhau mà mỗi giá trị xuất hiện
# ít nhất 50 lần. Ngoài đoạn đó mỗi giá trị chỉ xuất hiện vài lần, nên là nhiễu (notebook 1, mục 1.5).
DOMAIN = {
    'Age': (14, 56),
    'Num_Bank_Accounts': (0, 10),
    'Num_Credit_Card': (1, 10),
    'Interest_Rate': (1, 34),
    'Num_of_Loan': (0, 9),
    'Num_of_Delayed_Payment': (0, 28),
    'Num_Credit_Inquiries': (0, 17),
}
EMI_MAX = 1800        # EMI thật lớn nhất trong dữ liệu là 1779, những số lớn hơn là nhiễu
INCOME_REL_TOL = 0.5  # thu nhập năm lệch hơn 50% so với trung vị của chính khách đó thì coi là nhiễu

# Mỗi cột được điền theo cách nào (chọn bằng thí nghiệm che 10% giá trị ở notebook 2)
REPAIR_PLAN = {
    'cust_mode':   ['SSN', 'Occupation', 'Name', 'Credit_Mix'],
    'cust_median': ['Annual_Income', 'Interest_Rate', 'Outstanding_Debt',
                    'Num_of_Delayed_Payment', 'Changed_Credit_Limit'],
    'nearest':     ['Num_Bank_Accounts', 'Num_Credit_Card', 'Age', 'Total_EMI_per_month',
                    'Monthly_Inhand_Salary', 'Num_Credit_Inquiries'],
    'model_ratio': ['Amount_invested_monthly', 'Monthly_Balance'],
}

# Các cột đưa vào mô hình điền thiếu (không có Credit_Score)
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


# ---------------------------------------------------------------------------
# Chuẩn bị: số thứ tự tháng
# ---------------------------------------------------------------------------
def add_month_num(d):
    """Thêm cột Month_Num (1 đến 8). File đã sắp theo (Customer_ID, Month) nên cột này cũng là thứ tự thời gian."""
    d = d.copy()
    d['Month_Num'] = d['Month'].map({m: i + 1 for i, m in enumerate(MONTHS)}).astype(int)
    return d


# ---------------------------------------------------------------------------
# Bước 1: giá trị giả -> NaN
# ---------------------------------------------------------------------------
def mark_placeholders(d, log=None):
    """Đổi các giá trị giả sang NaN. Nếu có `log` (dict) thì ghi số ô đã đổi của từng cột."""
    d = d.copy()
    for col, token in PLACEHOLDERS.items():
        is_fake = d[col] == token
        if log is not None:
            log.setdefault('placeholder', {})[col] = int(is_fake.sum())
        d.loc[is_fake, col] = np.nan
    return d


# ---------------------------------------------------------------------------
# Bước 2: bỏ dấu "_" và ép kiểu số
# ---------------------------------------------------------------------------
def strip_numeric_text(d):
    """Bỏ dấu "_" ở đầu/cuối rồi ép sang số, ví dụ "24_" thành 24 và "14388.79_" thành 14388.79."""
    d = d.copy()
    for col in NUMERIC_TEXT_COLS:
        text = d[col].astype('string').str.replace(r'^_+|_+$', '', regex=True)
        d[col] = pd.to_numeric(text, errors='coerce').astype('float64')  # float64 để còn chứa được NaN
    return d


# ---------------------------------------------------------------------------
# Bước 3: giá trị vô lý -> NaN
# ---------------------------------------------------------------------------
def mark_invalid(d, log=None):
    """Đổi sang NaN những giá trị không thể là dữ liệu thật:
    - cột đếm nằm ngoài miền DOMAIN
    - Total_EMI_per_month quá lớn, hoặc là số nguyên dương (EMI thật là số lẻ dài, nên số tròn là nhiễu)
    - Annual_Income lệch hơn 50% so với trung vị của chính khách đó (thu nhập năm không đổi trong 8 tháng)
    """
    d = d.copy()
    log = {} if log is None else log
    invalid_count = log.setdefault('invalid', {})

    for col, (low, high) in DOMAIN.items():
        out_of_range = (d[col] < low) | (d[col] > high)
        invalid_count[col] = int(out_of_range.sum())
        d.loc[out_of_range, col] = np.nan

    emi = d['Total_EMI_per_month']
    bad_emi = (emi > EMI_MAX) | ((emi > 0) & (emi == emi.round(0)))
    invalid_count['Total_EMI_per_month'] = int(bad_emi.sum())
    d.loc[bad_emi, 'Total_EMI_per_month'] = np.nan

    median_income = d.groupby('Customer_ID')['Annual_Income'].transform('median')
    bad_income = (d['Annual_Income'] / median_income - 1).abs() > INCOME_REL_TOL
    invalid_count['Annual_Income'] = int(bad_income.sum())
    d.loc[bad_income, 'Annual_Income'] = np.nan
    return d


# ---------------------------------------------------------------------------
# Bước 4: điền lại các ô NaN
# Các hàm dưới đây dùng chung cho pipeline và cho thí nghiệm so sánh cách điền ở notebook 2.
# ---------------------------------------------------------------------------
def nearest_fill(d, col):
    """Điền ô thiếu bằng giá trị hợp lệ ở tháng gần nhất của cùng khách. Cách đều thì lấy tháng trước.
    Hợp với cột hay nhảy bậc (số tài khoản, lương, EMI) vì không kéo giá trị của giai đoạn này sang giai đoạn khác."""
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
    """Giá trị xuất hiện nhiều nhất của một Series (bỏ NaN). Không có giá trị nào thì trả về NaN."""
    modes = s.dropna().mode()
    return modes.iloc[0] if len(modes) else np.nan


def payment_behaviour_code(d):
    """Mã hóa Payment_Behaviour thành số (NaN thành -1) để làm đầu vào cho mô hình điền thiếu."""
    categories = sorted(d['Payment_Behaviour'].dropna().unique())
    return d['Payment_Behaviour'].map({c: i for i, c in enumerate(categories)}).fillna(-1)


def salary_for_scaling(d):
    """Lương tháng đã điền đủ, chặn dưới ở 1 để chia không bị lỗi. Dùng làm mẫu số cho các tỷ lệ so với lương."""
    return fill_numeric(d, 'Monthly_Inhand_Salary', 'nearest').clip(lower=1)


def model_fill(d, col, features=MODEL_IMPUTE_FEATURES, seed=0, ratio=False):
    """Điền cột `col` bằng HistGradientBoostingRegressor, học từ các cột khác của cùng dòng.
    Amount_invested_monthly và Monthly_Balance phụ thuộc nhiều vào lương và thói quen chi tiêu nên dùng cách này.

    ratio=True: mô hình dự đoán tỷ lệ so với lương (col / lương) rồi nhân lại với lương.
    Tiền đầu tư và số dư tăng theo thu nhập, nên tỷ lệ ổn định hơn giữa các mức lương. Ở thí nghiệm
    notebook 2 (mục 4.2), cách này sai ít hơn dự đoán thẳng số tiền ở mọi mức lương."""
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
    """Trả về cột số `col` đã điền theo `strategy`:
    global_median | cust_median | nearest | interpolate | model | model_ratio"""
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
    return filled.fillna(s.median())   # khách nào mất hết giá trị hợp lệ thì lấy trung vị chung


def fill_categorical(d, col, strategy):
    """Trả về cột phân loại `col` đã điền theo `strategy`: global_mode | cust_mode | nearest"""
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


def parse_history_months(x):
    """'22 Years and 9 Months' -> 273 (đổi ra tháng). Không đọc được thì trả về NaN."""
    m = re.match(r'(\d+) Years and (\d+) Months', str(x))
    return int(m.group(1)) * 12 + int(m.group(2)) if m else np.nan


def split_loan_types(x):
    """'Auto Loan, Credit-Builder Loan, and Home Equity Loan' -> ['Auto Loan', 'Credit-Builder Loan', 'Home Equity Loan']"""
    parts = re.split(r',\s*', str(x).replace(' and ', ' '))
    return [p.strip() for p in parts if p.strip()]


def count_loan_types(x):
    """Số loại khoản vay trong chuỗi Type_of_Loan. NaN hoặc 'No Loan' thì là 0."""
    if pd.isna(x) or x == 'No Loan':
        return 0
    return len(split_loan_types(x))


def fill_columns(d, cols, strategy, log):
    """Điền lần lượt các cột trong `cols` theo `strategy`, ghi số ô đã điền vào log."""
    for col in cols:
        log[col] = int(d[col].isna().sum())
        if strategy == 'cust_mode':
            d[col] = fill_categorical(d, col, strategy)
        else:
            d[col] = fill_numeric(d, col, strategy)


def rebuild_loan_count(d, log):
    """Num_of_Loan = số loại khoản vay ghi trong Type_of_Loan. Ở dữ liệu hợp lệ hai cột này khớp 100%,
    nên tính lại từ Type_of_Loan thay vì đoán."""
    log['Num_of_Loan'] = int(d['Num_of_Loan'].isna().sum())
    d['Num_of_Loan'] = d['Type_of_Loan'].map(count_loan_types)


def rebuild_credit_history(d, log):
    """Credit_History_Age tăng đúng 1 tháng mỗi tháng. Với mỗi khách, lấy (số tháng - thứ tự tháng) ở các
    tháng còn đọc được, lấy trung vị, rồi suy ra giá trị cho tháng bị mất. Kết quả lưu ở Credit_History_Months."""
    months = d['Credit_History_Age'].map(parse_history_months)
    offset = (months - d['Month_Num']).groupby(d['Customer_ID']).transform('median')
    log['Credit_History_Age'] = int(months.isna().sum())
    d['Credit_History_Months'] = months.fillna(offset + d['Month_Num']).fillna(months.median())
    d.drop(columns='Credit_History_Age', inplace=True)


def repair(d, plan=REPAIR_PLAN, log=None):
    """Điền lại các ô NaN. Thứ tự các bước có ý nghĩa, vì bước sau dùng kết quả của bước trước."""
    d = d.copy()
    log = {} if log is None else log
    repaired = log.setdefault('repaired', {})

    # Payment_Behaviour đổi ngẫu nhiên theo tháng, thí nghiệm ở notebook 2 cho thấy không cách điền nào
    # đoán tốt hơn ngẫu nhiên. Nên giữ lại thông tin "không rõ" thành một nhóm riêng, không bịa giá trị.
    repaired['Payment_Behaviour'] = int(d['Payment_Behaviour'].isna().sum())
    d['Payment_Behaviour'] = d['Payment_Behaviour'].fillna('Unknown')

    # Type_of_Loan trống nghĩa là khách không vay gì (trống cả 8 tháng và Num_of_Loan = 0, notebook 1 mục 1.3).
    # Đây là giá trị thật nên điền 'No Loan', không điền bằng loại vay phổ biến nhất.
    repaired['Type_of_Loan'] = int(d['Type_of_Loan'].isna().sum())
    d['Type_of_Loan'] = d['Type_of_Loan'].fillna('No Loan')

    fill_columns(d, plan['cust_mode'], 'cust_mode', repaired)      # cột cố định theo khách
    fill_columns(d, plan['cust_median'], 'cust_median', repaired)  # cột gần như không đổi theo khách
    fill_columns(d, plan['nearest'], 'nearest', repaired)          # cột nhảy bậc theo thời gian

    rebuild_loan_count(d, repaired)
    rebuild_credit_history(d, repaired)

    # Hai cột tiền phụ thuộc vào lương và chi tiêu: điền bằng mô hình, sau khi các cột còn lại đã đủ
    fill_columns(d, plan['model_ratio'], 'model_ratio', repaired)
    return d


# ---------------------------------------------------------------------------
# Hoàn thiện
# ---------------------------------------------------------------------------
def finalize(d):
    """Ép các cột đếm về số nguyên và sắp lại thứ tự cột."""
    d = d.copy()
    for col in INT_COLS:
        d[col] = d[col].round().astype(int)
    return d[[c for c in FINAL_COLUMN_ORDER if c in d.columns]]


def clean(df_raw, return_log=False):
    """Chạy cả pipeline: train.csv đọc thô -> dữ liệu sạch. return_log=True thì trả thêm dict thống kê."""
    log = {}
    d = add_month_num(df_raw)
    d = mark_placeholders(d, log)
    d = strip_numeric_text(d)
    d = mark_invalid(d, log)
    d = repair(d, log=log)
    d = finalize(d)
    return (d, log) if return_log else d


# ---------------------------------------------------------------------------
# Kiểm tra kết quả
# ---------------------------------------------------------------------------
def validate_clean(d):
    """12 kiểm tra ràng buộc trên dữ liệu đã sạch. Trả về dict {tên kiểm tra: True/False}."""
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
