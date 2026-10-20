"""
Dự đoán test.csv và xuất file nộp bài (submission.csv).

Cấu trúc dữ liệu của cuộc thi: test.csv KHÔNG phải khách hàng mới. Đó là 4 tháng tiếp theo (September đến
December) của đúng 12.500 khách đã có trong train.csv (January đến August), ID dòng không trùng nhau.
Vì vậy:
  - Tháng 9 đến 12 nối tiếp tháng 1 đến 8 của cùng khách. Các cột "đổi theo thời gian" (Credit_History_Age,
    Age, các đặc trưng LAG) nhìn được lịch sử trong train, đúng như khi dự đoán thật: biết quá khứ, chưa biết tương lai.
  - Số liệu CV ở notebook 4 chia theo khách (khách mới hoàn toàn). Còn test.csv là cùng khách ở giai đoạn sau,
    nên đó là hai tình huống khác nhau. `check_robustness.py` đo thêm tình huống thứ hai bằng cách chia theo thời gian.

Các hàm làm sạch và tạo đặc trưng lấy lại nguyên từ credit_cleaning.py và credit_features.py.
Không hàm nào ở đây đọc cột nhãn Credit_Score của test (test.csv cũng không có cột này).
"""
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import credit_cleaning as cc
import credit_features as cf

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------
ALL_MONTHS = cc.MONTHS + ['September', 'October', 'November', 'December']   # thứ tự thời gian thật
MAIN_GROUPS = ['BASE', 'RATIO', 'LOAN', 'PB_SPLIT', 'LAG']                  # bộ đặc trưng chính của notebook 4
LABEL_TO_CODE = {'Poor': 0, 'Standard': 1, 'Good': 2}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}
IDENTIFIER_COLS = ['ID', 'Customer_ID', 'Name', 'SSN']   # không bao giờ là đặc trưng (kiểm tra ở assert_no_leak_columns)


# ---------------------------------------------------------------------------
# Đọc dữ liệu và kiểm tra cấu trúc
# ---------------------------------------------------------------------------
def load_raw(train_path, test_path):
    """Đọc train.csv và test.csv ở dạng chữ nguyên bản, rồi kiểm tra cấu trúc hai file.
    low_memory=False để pandas không đoán kiểu cột theo từng khối (cột lẫn số và chữ bẩn sẽ bị đoán lệch)."""
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    assert list(train.columns) == list(test.columns) + [cc.TARGET], 'test.csv phải có đúng các cột của train.csv trừ Credit_Score'
    assert test['ID'].is_unique and train['ID'].is_unique, 'ID bị trùng'
    assert set(train['ID']).isdisjoint(set(test['ID'])), 'ID của test trùng với train'
    assert set(train['Month']) == set(cc.MONTHS) and set(test['Month']) == set(ALL_MONTHS[8:]), \
        'tháng của train/test không đúng như dự kiến'
    assert set(test['Customer_ID']) <= set(train['Customer_ID']), 'test có khách không thấy trong train'
    return train, test


def describe_structure(train, test):
    """Các con số cấu trúc để ghi vào nhật ký chạy."""
    return {'train_rows': len(train), 'test_rows': len(test),
            'train_customers': int(train['Customer_ID'].nunique()),
            'test_customers': int(test['Customer_ID'].nunique()),
            'test_customers_also_in_train': int(len(set(test['Customer_ID']) & set(train['Customer_ID']))),
            'train_months': cc.MONTHS, 'test_months': ALL_MONTHS[8:],
            'rows_per_customer_train': sorted(train.groupby('Customer_ID').size().unique().tolist()),
            'rows_per_customer_test': sorted(test.groupby('Customer_ID').size().unique().tolist())}


# ---------------------------------------------------------------------------
# Làm sạch chuỗi 12 tháng
# ---------------------------------------------------------------------------
def build_panel(train, test):
    """Nối train (không lấy nhãn) rồi test, giữ nguyên thứ tự dòng của hai file. Trong mỗi file, các tháng của
    một khách đã nằm đúng thứ tự thời gian, và tháng của test đều sau tháng của train, nên sau khi nối thì mọi khách
    có đủ 12 tháng theo thứ tự (hàm main kiểm tra điều này). Giữ nguyên thứ tự dòng nghĩa là dòng test vẫn nằm cuối
    và theo đúng thứ tự của test.csv.
    Nhãn Credit_Score bị bỏ ngay ở đây nên không thể lọt vào bước làm sạch hay tạo đặc trưng nào."""
    panel = pd.concat([train.drop(columns=cc.TARGET), test], ignore_index=True)
    panel['Month_Num'] = panel['Month'].map({m: i + 1 for i, m in enumerate(ALL_MONTHS)}).astype(int)
    assert cc.TARGET not in panel.columns
    return panel


def clean_panel(panel_raw, return_log=False):
    """Chạy lại đúng các bước của cc.clean() nhưng trên chuỗi 12 tháng. cc.clean() chỉ biết 8 tháng nên không
    dùng trực tiếp được. Các bước, thứ tự và tham số giữ nguyên, chỉ khác cách đánh số tháng (đã có sẵn Month_Num).
    Khi đưa vào chuỗi 8 tháng thì kết quả phải giống hệt cc.clean() (kiểm tra ở check_robustness.py)."""
    log = {}
    d = panel_raw.copy()
    d = cc.mark_placeholders(d, log)
    d = cc.strip_numeric_text(d)
    d = cc.mark_invalid(d, log)
    marked = d.copy()                      # trước khi điền, để đếm số ô phải sửa theo từng cột
    d = cc.repair(d, log=log)
    d = cc.finalize(d)
    return (d, log, marked) if return_log else d


def validate_panel(d, n_months):
    """Các ràng buộc cho chuỗi sạch n_months tháng. Lấy lại các kiểm tra của cc.validate_clean và thay hai kiểm tra
    gắn với con số 8 tháng bằng bản tổng quát."""
    checks = cc.validate_clean(d)
    for name in [k for k in checks if k.startswith('8 dòng') or k.startswith('Tuổi trong 1 khách')]:
        checks.pop(name)
    by_cust = d.groupby('Customer_ID')
    checks[f'{n_months} dòng / khách, đúng thứ tự tháng'] = bool(
        by_cust['Month_Num'].apply(lambda s: list(s) == list(range(1, n_months + 1))).all())
    checks['Tuổi trong 1 khách chênh tối đa 1'] = bool((by_cust['Age'].max() - by_cust['Age'].min()).le(1).all())
    return checks


# ---------------------------------------------------------------------------
# Ma trận đặc trưng
# ---------------------------------------------------------------------------
def assert_no_leak_columns(X):
    """Ma trận đưa vào mô hình không được chứa nhãn hay cột định danh."""
    bad = [c for c in X.columns if c == cc.TARGET or any(c == i or c.startswith(i + '_') for i in IDENTIFIER_COLS)]
    assert not bad, f'có cột không được dùng làm đặc trưng: {bad}'


def feature_matrix(feat, groups=MAIN_GROUPS):
    """Bảng đặc trưng -> ma trận số của các nhóm đã chọn."""
    X = cf.make_matrix(feat, groups)
    assert_no_leak_columns(X)
    assert not X.isna().any().any(), 'ma trận đặc trưng còn ô thiếu'
    return X


def align_columns(X, reference_columns):
    """Đưa ma trận về đúng các cột (và thứ tự cột) của ma trận lúc huấn luyện. Cột one-hot nào không có trong tập
    cần dự đoán thì điền 0. Nếu có cột lạ (một nhóm phân loại chưa từng thấy khi huấn luyện) thì dừng lại, không
    âm thầm bỏ đi."""
    extra = set(X.columns) - set(reference_columns)
    assert not extra, f'có cột chưa từng thấy khi huấn luyện: {sorted(extra)}'
    return X.reindex(columns=list(reference_columns), fill_value=0.0)


# ---------------------------------------------------------------------------
# Mô hình
# ---------------------------------------------------------------------------
def load_chosen_model(metrics_path):
    """Đọc mô hình và siêu tham số đã chọn ở notebook 4 (data/reports/final_test_metrics.json).
    Notebook đã chọn bằng CV theo khách, không dùng tập kiểm tra, nên script này không tự chọn lại bằng mắt."""
    with open(metrics_path, encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics['model'], metrics['params'], metrics['class_weight']


def make_model(name, params, class_weight, seed=0, n_jobs=2):
    """Tạo mô hình chưa huấn luyện từ tên và siêu tham số đã chọn."""
    if name.startswith('Random Forest'):
        return RandomForestClassifier(n_jobs=n_jobs, random_state=seed, class_weight=class_weight, **params)
    if name.startswith('HistGradientBoosting'):
        return HistGradientBoostingClassifier(random_state=seed, class_weight=class_weight, **params)
    raise ValueError(f'không biết mô hình {name!r}')


def encode_labels(y):
    """'Poor', 'Standard', 'Good' -> 0, 1, 2. Báo lỗi nếu có nhãn lạ."""
    codes = y.map(LABEL_TO_CODE)
    assert codes.notna().all(), 'có nhãn ngoài Poor/Standard/Good'
    return codes.astype(int)


def decode_labels(codes):
    return pd.Series(codes).map(CODE_TO_LABEL).to_numpy()


# ---------------------------------------------------------------------------
# File nộp bài
# ---------------------------------------------------------------------------
def build_submission(test_raw, predicted_labels):
    """Bảng 2 cột (ID, Credit_Score), đúng thứ tự dòng của test.csv. ID lấy nguyên văn từ test.csv."""
    return pd.DataFrame({'ID': test_raw['ID'].to_numpy(), 'Credit_Score': predicted_labels})


def check_submission(path, test_raw):
    """Đọc lại file đã ghi ra đĩa và kiểm tra. Trả về dict {kiểm tra: True/False}, và dừng chương trình nếu có kiểm tra sai."""
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    raw_text = open(path, encoding='utf-8').read().splitlines()
    checks = {
        'Đúng 2 cột: ID, Credit_Score': list(sub.columns) == ['ID', 'Credit_Score'],
        'Dòng đầu là tiêu đề, không có cột index': raw_text[0] == 'ID,Credit_Score',
        'Đúng số dòng của test.csv': len(sub) == len(test_raw),
        'ID giống test.csv, cùng thứ tự': sub['ID'].tolist() == test_raw['ID'].tolist(),
        'Không mất ID, không trùng ID': sub['ID'].is_unique and set(sub['ID']) == set(test_raw['ID']),
        'Nhãn chỉ gồm Poor / Standard / Good': set(sub['Credit_Score']) <= set(LABEL_TO_CODE),
        'Không có ô trống': bool((sub != '').all().all()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f'file nộp bài không đạt: {failed}'
    return checks
