"""Các hàm cho bước dự đoán test.csv và ghi file output. Test là 4 tháng sau của chính 12.500 khách trong train."""
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

import credit_cleaning as cc
import credit_features as cf

ALL_MONTHS = cc.MONTHS + ['September', 'October', 'November', 'December']
MAIN_GROUPS = ['BASE', 'RATIO', 'LOAN', 'PB_SPLIT', 'LAG']
LABEL_TO_CODE = {'Poor': 0, 'Standard': 1, 'Good': 2}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}
IDENTIFIER_COLS = ['ID', 'Customer_ID', 'Name', 'SSN']


def load_raw(train_path, test_path):
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
    return {'train_rows': len(train), 'test_rows': len(test),
            'train_customers': int(train['Customer_ID'].nunique()),
            'test_customers': int(test['Customer_ID'].nunique()),
            'test_customers_also_in_train': int(len(set(test['Customer_ID']) & set(train['Customer_ID']))),
            'train_months': cc.MONTHS, 'test_months': ALL_MONTHS[8:],
            'rows_per_customer_train': sorted(train.groupby('Customer_ID').size().unique().tolist()),
            'rows_per_customer_test': sorted(test.groupby('Customer_ID').size().unique().tolist())}


# tỷ lệ dòng có giá trị giả và ngoài miền ở từng cột (bỏ Annual_Income vì luật của cột này phụ thuộc số tháng)
def noise_rates(raw):
    log = {}
    d = cc.strip_numeric_text(cc.mark_placeholders(raw, log))
    cc.mark_invalid(d, log)
    rates = {}
    for kind, name in (('placeholder', 'giá trị giả'), ('invalid', 'ngoài miền hợp lệ')):
        for col, n in log[kind].items():
            if col != 'Annual_Income':
                rates.setdefault(col, {})[name] = n / len(raw)
    return pd.DataFrame(rates).T.fillna(0.0)


def credit_history_breaks(panel):
    months = panel['Credit_History_Age'].map(cc.parse_history_months)
    offset = months - panel['Month_Num']
    return int((offset.groupby(panel['Customer_ID']).nunique() > 1).sum())


# nối train với test, bỏ nhãn ngay ở đây để không lọt vào bước nào phía sau
def build_panel(train, test):
    panel = pd.concat([train.drop(columns=cc.TARGET), test], ignore_index=True)
    panel['Month_Num'] = panel['Month'].map({m: i + 1 for i, m in enumerate(ALL_MONTHS)}).astype(int)
    assert cc.TARGET not in panel.columns
    return panel


# giống cc.clean() nhưng chạy trên 12 tháng
def clean_panel(panel_raw, return_log=False):
    log = {}
    d = panel_raw.copy()
    d = cc.mark_placeholders(d, log)
    d = cc.strip_numeric_text(d)
    d = cc.mark_invalid(d, log)
    marked = d.copy()
    d = cc.repair(d, log=log)
    d = cc.finalize(d)
    return (d, log, marked) if return_log else d


def validate_panel(d, n_months):
    checks = cc.validate_clean(d)
    for name in [k for k in checks if k.startswith('8 dòng') or k.startswith('Tuổi trong 1 khách')]:
        checks.pop(name)
    by_cust = d.groupby('Customer_ID')
    checks[f'{n_months} dòng / khách, đúng thứ tự tháng'] = bool(
        by_cust['Month_Num'].apply(lambda s: list(s) == list(range(1, n_months + 1))).all())
    checks['Tuổi trong 1 khách chênh tối đa 1'] = bool((by_cust['Age'].max() - by_cust['Age'].min()).le(1).all())
    return checks


def assert_no_leak_columns(X):
    bad = [c for c in X.columns if c == cc.TARGET or any(c == i or c.startswith(i + '_') for i in IDENTIFIER_COLS)]
    assert not bad, f'có cột không được dùng làm đặc trưng: {bad}'


def feature_matrix(feat, groups=MAIN_GROUPS):
    X = cf.make_matrix(feat, groups)
    assert_no_leak_columns(X)
    assert not X.isna().any().any(), 'ma trận đặc trưng còn ô thiếu'
    return X


def align_columns(X, reference_columns):
    extra = set(X.columns) - set(reference_columns)
    assert not extra, f'có cột chưa từng thấy khi huấn luyện: {sorted(extra)}'
    return X.reindex(columns=list(reference_columns), fill_value=0.0)


def load_chosen_model(metrics_path):
    with open(metrics_path, encoding='utf-8') as f:
        metrics = json.load(f)
    return metrics['model'], metrics['params'], metrics['class_weight']


def make_model(name, params, class_weight, seed=0, n_jobs=2):
    if name.startswith('Random Forest'):
        return RandomForestClassifier(n_jobs=n_jobs, random_state=seed, class_weight=class_weight, **params)
    if name.startswith('HistGradientBoosting'):
        return HistGradientBoostingClassifier(random_state=seed, class_weight=class_weight, **params)
    raise ValueError(f'không biết mô hình {name!r}')


def encode_labels(y):
    codes = y.map(LABEL_TO_CODE)
    assert codes.notna().all(), 'có nhãn ngoài Poor/Standard/Good'
    return codes.astype(int)


def decode_labels(codes):
    return pd.Series(codes).map(CODE_TO_LABEL).to_numpy()


def build_output(test_raw, predicted_labels):
    return pd.DataFrame({'ID': test_raw['ID'].to_numpy(), 'Credit_Score': predicted_labels})


def check_output(path, test_raw):
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
    assert not failed, f'file output không đạt: {failed}'
    return checks
