"""
Tạo đặc trưng (feature) từ dữ liệu đã làm sạch, tức kết quả của credit_cleaning.clean().
Notebook 3 dùng để phân tích đặc trưng, notebook 4 dùng để huấn luyện mô hình.

Các nhóm đặc trưng:
    BASE      các cột gốc sau khi làm sạch (số và phân loại)
    RATIO     tỷ số tài chính tính trong từng dòng (nợ / thu nhập, EMI / lương, ...)
    LOAN      cột 0/1 cho từng loại khoản vay
    PB_SPLIT  tách Payment_Behaviour thành mức chi tiêu và độ lớn khoản thanh toán
    LAG       đặc trưng theo thời gian, chỉ dùng tháng hiện tại và các tháng trước đó
    CUST_AGG  thống kê trên cả 8 tháng của khách (có dùng cả các tháng sau, xem ghi chú ở hàm bên dưới)

Không dùng cột nhãn Credit_Score để tạo bất kỳ đặc trưng nào.
"""
import pandas as pd
from credit_cleaning import split_loan_types

# ---------------------------------------------------------------------------
# Danh sách cột của từng nhóm
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Từng nhóm đặc trưng
# ---------------------------------------------------------------------------
def add_ratio_features(d):
    """Các tỷ số tài chính trong từng dòng. Mẫu số được chặn dưới ở 1 để không chia cho 0."""
    d = d.copy()
    salary = d['Monthly_Inhand_Salary'].clip(lower=1)
    d['Debt_to_Income'] = d['Outstanding_Debt'] / d['Annual_Income']
    d['EMI_to_Salary'] = d['Total_EMI_per_month'] / salary
    d['Invest_to_Salary'] = d['Amount_invested_monthly'] / salary
    d['Balance_to_Salary'] = d['Monthly_Balance'] / salary
    # phần lương còn lại sau khi trừ EMI, tiền đầu tư và số dư cuối tháng, coi như tiền đã chi tiêu
    d['Implied_Spend_Ratio'] = (salary - d['Total_EMI_per_month'] - d['Amount_invested_monthly']
                                - d['Monthly_Balance']) / salary
    d['Debt_per_Loan'] = d['Outstanding_Debt'] / d['Num_of_Loan'].clip(lower=1)
    d['Has_Loan'] = (d['Num_of_Loan'] > 0).astype(int)
    return d


def add_loan_flags(d):
    """Mỗi loại khoản vay thành một cột 0/1 (khách có loại vay đó hay không)."""
    d = d.copy()
    types_of = {u: set(split_loan_types(u)) for u in d['Type_of_Loan'].unique()}
    for loan_type, col in zip(LOAN_TYPES, LOAN):
        d[col] = d['Type_of_Loan'].map(lambda u: int(loan_type in types_of[u]))
    return d


def add_payment_split(d):
    """Tách Payment_Behaviour ('Low_spent_Small_value_payments') thành hai cột: mức chi tiêu và độ lớn khoản trả.
    Giá trị Unknown được giữ nguyên ở cả hai cột."""
    d = d.copy()
    d['Spend_Level'] = d['Payment_Behaviour'].str.extract(r'^(High|Low)_spent')[0].fillna('Unknown')
    d['Payment_Size'] = d['Payment_Behaviour'].str.extract(r'_(Small|Medium|Large)_value')[0].fillna('Unknown')
    return d


def add_lag_features(d):
    """Đặc trưng theo thời gian, chỉ nhìn tháng hiện tại và các tháng trước nên không dùng thông tin tương lai:
    giá trị tháng trước (_prev), chênh lệch so với tháng trước (_delta), trung bình từ tháng đầu đến tháng hiện tại (_expmean).
    Tháng đầu tiên chưa có tháng trước nên lấy chính giá trị của nó."""
    d = d.copy()
    by_cust = d.groupby('Customer_ID')
    for col in LAG_COLS:
        d[f'{col}_prev'] = by_cust[col].shift(1).fillna(d[col])
        d[f'{col}_delta'] = d[col] - d[f'{col}_prev']
        d[f'{col}_expmean'] = by_cust[col].cumsum() / d['Month_Num']
    return d


def add_customer_features(d):
    """Thống kê trên cả 8 tháng của mỗi khách: trung bình, độ lệch chuẩn, và tỷ lệ tháng không trả mức tối thiểu.
    Nhóm này dùng cả dữ liệu của các tháng sau tháng đang xét, nên chỉ hợp lý khi đã biết đủ 8 tháng.
    Vì vậy ở notebook 4 nhóm này chỉ để so sánh, mô hình chính không dùng."""
    d = d.copy()
    by_cust = d.groupby('Customer_ID')
    for col in AGG_COLS:
        d[f'{col}_cust_mean'] = by_cust[col].transform('mean')
        d[f'{col}_cust_std'] = by_cust[col].transform('std')
    d['MinPay_No_share'] = (d['Payment_of_Min_Amount'] == 'No').groupby(d['Customer_ID']).transform('mean')
    return d


def build_features(clean_df):
    """Dữ liệu đã làm sạch -> bảng có thêm đủ các nhóm đặc trưng."""
    d = add_ratio_features(clean_df)
    d = add_loan_flags(d)
    d = add_payment_split(d)
    d = add_lag_features(d)
    d = add_customer_features(d)
    return d


def make_matrix(d, groups=('BASE',)):
    """Ma trận đặc trưng dạng số từ các nhóm đã chọn (cột phân loại được one-hot)."""
    cols = [c for g in groups for c in FEATURE_GROUPS[g]]
    X = d[cols].copy()
    categorical = [c for c in cols if c in CATEGORICAL]
    X = pd.get_dummies(X, columns=categorical, dtype=float)
    return X.astype(float)
