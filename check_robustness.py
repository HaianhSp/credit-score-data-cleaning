"""
Các kiểm tra bổ sung cho pipeline, tương ứng với các yêu cầu kiểm soát chất lượng thực nghiệm:

  folds        không khách nào nằm ở cả train và validation của mỗi phần CV, và các phần có giữ được tỷ lệ nhãn không
  regression   hàm làm sạch 12 tháng (credit_submission.clean_panel) cho kết quả giống hệt cc.clean() trên 8 tháng
  imputer      mô hình điền thiếu của hai cột tiền (HistGradientBoosting) là bước duy nhất của phần làm sạch học
               xuyên khách hàng. Ở pipeline chính nó được học trên toàn bộ train (không dùng nhãn). Ở đây học lại chỉ
               trên phần train của từng phần CV rồi so macro-F1 với cách thường
  features     ba đặc trưng tài chính trong đề (Monthly_Gross_Income, Debt_to_Income_Ratio, Monthly_Savings)
               có làm mô hình tốt hơn không. Quy tắc đặt trước khi chạy: chỉ đưa vào bộ chính nếu macro-F1 cao hơn
               ở cả 5 phần CV
  timesplit    test.csv là các tháng sau của cùng khách, còn CV chia theo khách. Ở đây giả lập đúng tình huống test:
               huấn luyện trên tháng 1 đến 4, dự đoán tháng 5 đến 8 (có nhãn nên chấm điểm được)

Cách chạy, từ thư mục gốc của dự án (cần data/train.csv, data/credit_clean.csv, data/credit_features.csv):
    python3 check_robustness.py                       # chạy tất cả, khoảng 30 phút
    python3 check_robustness.py --checks folds,timesplit
Kết quả ghi vào data/reports/robustness_checks.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import credit_cleaning as cc            # noqa: E402
import credit_features as cf            # noqa: E402
import credit_submission as cs          # noqa: E402

DATA = ROOT / 'data'
CLASSES = ['Poor', 'Standard', 'Good']
CLASS_CODES = [cs.LABEL_TO_CODE[c] for c in CLASSES]
ALL_CHECKS = ['folds', 'regression', 'imputer', 'features', 'timesplit']


# ---------------------------------------------------------------------------
# Dữ liệu và cách chia giống hệt notebook 4
# ---------------------------------------------------------------------------
def load_inputs():
    """Đọc dữ liệu và dựng lại đúng cách chia của notebook 4 (cùng seed, cùng tham số)."""
    raw = pd.read_csv(DATA / 'train.csv', low_memory=False)
    clean = pd.read_csv(DATA / 'credit_clean.csv')
    feat = pd.read_csv(DATA / 'credit_features.csv')
    assert (raw['ID'].to_numpy() == clean['ID'].to_numpy()).all() and (clean['ID'].to_numpy() == feat['ID'].to_numpy()).all()
    y_text, customer = clean[cc.TARGET], clean['Customer_ID']

    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(clean, y_text, customer))
    y_train_text = y_text.iloc[train_idx].reset_index(drop=True)
    cust_train = customer.iloc[train_idx].reset_index(drop=True)
    # nhãn dạng chữ để chia giống notebook 4 (thứ tự lớp khi chia phụ thuộc kiểu nhãn)
    folds5 = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                  .split(np.zeros(len(y_train_text)), y_train_text, cust_train))
    folds3 = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=7)
                  .split(np.zeros(len(y_train_text)), y_train_text, cust_train))
    return dict(raw=raw, clean=clean, feat=feat, train_idx=train_idx, test_idx=test_idx, y_train_text=y_train_text,
                y_train=cs.encode_labels(y_train_text), cust_train=cust_train, folds5=folds5, folds3=folds3)


def final_model_factory():
    """Mô hình cuối đã chọn ở notebook 4, dưới dạng hàm tạo mô hình mới."""
    name, params, class_weight = cs.load_chosen_model(DATA / 'reports' / 'final_test_metrics.json')
    return name, (lambda: cs.make_model(name, params, class_weight, seed=0))


def fold_scores(make_model, X, y, folds):
    """CV theo `folds` (danh sách (dòng huấn luyện, dòng validation) theo vị trí trong X). Trả về từng phần:
    accuracy, macro-F1, precision/recall/F1 của từng lớp."""
    rows = []
    for fit_rows, val_rows in folds:
        model = make_model().fit(X.iloc[fit_rows], y.iloc[fit_rows])
        pred = model.predict(X.iloc[val_rows])
        p, r, f, _ = precision_recall_fscore_support(y.iloc[val_rows], pred, labels=CLASS_CODES, zero_division=0)
        rows.append({'accuracy': accuracy_score(y.iloc[val_rows], pred),
                     'macro_f1': f1_score(y.iloc[val_rows], pred, average='macro'),
                     **{f'precision_{c}': v for c, v in zip(CLASSES, p)},
                     **{f'recall_{c}': v for c, v in zip(CLASSES, r)},
                     **{f'f1_{c}': v for c, v in zip(CLASSES, f)}})
    return pd.DataFrame(rows)


def summarize(scores):
    """Bảng điểm từng phần -> {chỉ số: {mean, std}}."""
    return {col: {'mean': float(scores[col].mean()), 'std': float(scores[col].std(ddof=0))} for col in scores.columns}


def say(text):
    print(text, flush=True)


# ---------------------------------------------------------------------------
# Kiểm tra 1: cách chia CV
# ---------------------------------------------------------------------------
def check_folds(inp):
    """Mỗi phần CV: tập khách của train và validation phải rời nhau; ghi tỷ lệ nhãn từng phần để biết có giữ được không."""
    overall = inp['y_train_text'].value_counts(normalize=True)
    out = {}
    for name, folds in (('cv5', inp['folds5']), ('cv3 (chỉnh tham số)', inp['folds3'])):
        worst, details = 0.0, []
        for k, (fit_rows, val_rows) in enumerate(folds, 1):
            shared = set(inp['cust_train'].iloc[fit_rows]) & set(inp['cust_train'].iloc[val_rows])
            assert not shared, f'{name} phần {k}: {len(shared)} khách nằm ở cả train và validation'
            share = inp['y_train_text'].iloc[val_rows].value_counts(normalize=True)
            gap = float((share - overall).abs().max())
            worst = max(worst, gap)
            details.append({'phần': k, 'khách validation': int(inp['cust_train'].iloc[val_rows].nunique()),
                            'lệch tỷ lệ nhãn lớn nhất so với toàn bộ (điểm %)': round(100 * gap, 2)})
        out[name] = {'không khách nào trùng giữa train và validation': True, 'các phần': details,
                     'lệch tỷ lệ nhãn lớn nhất (điểm %)': round(100 * worst, 2)}
        say(f'[folds] {name}: không có khách nào nằm ở cả train và validation; tỷ lệ nhãn từng phần lệch tối đa '
            f'{100 * worst:.2f} điểm % so với toàn bộ')
    return out


# ---------------------------------------------------------------------------
# Kiểm tra 2: hàm làm sạch 12 tháng khớp với cc.clean() trên 8 tháng
# ---------------------------------------------------------------------------
def check_regression(inp):
    raw = inp['raw']
    expected = cc.clean(raw).drop(columns=cc.TARGET)
    got = cs.clean_panel(cc.add_month_num(raw.drop(columns=cc.TARGET)))
    identical = expected.equals(got)
    differing = 0 if identical else int((expected.ne(got) & ~(expected.isna() & got.isna())).sum().sum())
    say(f'[regression] clean_panel (8 tháng) so với cc.clean(): giống hệt = {identical}, số ô khác = {differing}')
    assert identical, 'clean_panel khác cc.clean() trên cùng dữ liệu 8 tháng'
    return {'giống hệt cc.clean() trên 8 tháng': identical, 'số ô khác': differing}


# ---------------------------------------------------------------------------
# Kiểm tra 3: học lại mô hình điền thiếu chỉ trên phần train của từng phần CV
# ---------------------------------------------------------------------------
def prepare_before_money_fill(raw):
    """Chạy pipeline làm sạch đến ngay trước bước điền hai cột tiền bằng mô hình (plan model_ratio rỗng)."""
    d = cc.add_month_num(raw.drop(columns=cc.TARGET))
    d = cc.mark_placeholders(d)
    d = cc.strip_numeric_text(d)
    d = cc.mark_invalid(d)
    return cc.repair(d, plan={**cc.REPAIR_PLAN, 'model_ratio': []})


def fill_money_columns(d_before, fit_rows=None):
    """Điền Amount_invested_monthly và Monthly_Balance bằng cc.model_fill. fit_rows (mảng bool theo dòng) cho biết mô
    hình chỉ được học từ những dòng nào. Cách làm: che giá trị của cột đang điền ở các dòng ngoài fit_rows để
    cc.model_fill không học từ chúng, rồi chỉ lấy dự đoán cho các ô vốn bị thiếu. fit_rows=None là cách của pipeline chính."""
    d = d_before.copy()
    use = pd.Series(True, index=d.index) if fit_rows is None else pd.Series(fit_rows, index=d.index)
    for col in cc.REPAIR_PLAN['model_ratio']:
        predicted = cc.model_fill(d.assign(**{col: d[col].where(use)}), col, ratio=True)
        d[col] = d[col].where(d[col].notna(), predicted).fillna(d.loc[use, col].median())
    return cc.finalize(d)


def check_imputer(inp, make_model):
    raw, y, train_idx = inp['raw'], inp['y_train'], inp['train_idx']
    d_before = prepare_before_money_fill(raw)

    t = time.time()
    clean_all = fill_money_columns(d_before)                      # cách của pipeline chính: học trên mọi dòng
    same_as_pipeline = clean_all.equals(cc.clean(raw).drop(columns=cc.TARGET))
    say(f'[imputer] dựng lại pipeline chính từng bước: giống hệt cc.clean() = {same_as_pipeline}')
    assert same_as_pipeline
    X_all = cs.feature_matrix(cf.build_features(clean_all)).iloc[train_idx].reset_index(drop=True)
    standard = fold_scores(make_model, X_all, y, inp['folds5'])
    say(f'[imputer] học trên mọi dòng      : macro-F1 {standard.macro_f1.mean():.4f} ± {standard.macro_f1.std(ddof=0):.4f}'
        f' ({time.time() - t:.0f}s)')

    refit_rows = []
    for k, (fit_rows, val_rows) in enumerate(inp['folds5'], 1):
        fit_positions = train_idx[fit_rows]                        # vị trí trong bảng 100.000 dòng
        mask = np.zeros(len(raw), dtype=bool)
        mask[fit_positions] = True
        clean_k = fill_money_columns(d_before, fit_rows=mask)
        X_k = cs.feature_matrix(cf.build_features(clean_k))
        model = make_model().fit(X_k.iloc[fit_positions], y.iloc[fit_rows])
        pred = model.predict(X_k.iloc[train_idx[val_rows]])
        refit_rows.append({'accuracy': accuracy_score(y.iloc[val_rows], pred),
                           'macro_f1': f1_score(y.iloc[val_rows], pred, average='macro')})
        say(f'[imputer] phần {k}: học lại trên phần train → macro-F1 {refit_rows[-1]["macro_f1"]:.4f}'
            f' (cách thường {standard.macro_f1.iloc[k - 1]:.4f})')
    refit = pd.DataFrame(refit_rows)
    diff = refit.macro_f1 - standard.macro_f1
    result = {'học trên mọi dòng': summarize(standard[['accuracy', 'macro_f1']]),
              'học lại trên phần train của từng phần CV': summarize(refit),
              'chênh macro-F1 từng phần (học lại trừ cách thường)': [round(float(v), 5) for v in diff],
              'chênh macro-F1 trung bình': round(float(diff.mean()), 5)}
    say(f'[imputer] chênh macro-F1 (học lại − cách thường): trung bình {diff.mean():+.4f}, '
        f'từng phần {[round(float(v), 4) for v in diff]}')
    return result


# ---------------------------------------------------------------------------
# Kiểm tra 4: ba đặc trưng tài chính trong đề
# ---------------------------------------------------------------------------
def spec_features(feat):
    """Ba đặc trưng đề yêu cầu. Debt_to_Income_Ratio gần như trùng Debt_to_Income đã có (chỉ khác 1e-5 ở mẫu số)."""
    return pd.DataFrame({
        'Monthly_Gross_Income': feat['Annual_Income'] / 12,
        'Debt_to_Income_Ratio': feat['Outstanding_Debt'] / (feat['Annual_Income'] + 1e-5),
        'Monthly_Savings': feat['Monthly_Inhand_Salary'] - feat['Amount_invested_monthly']})


def check_features(inp, make_model):
    feat, train_idx, y = inp['feat'], inp['train_idx'], inp['y_train']
    X_main = cs.feature_matrix(feat)
    extra = spec_features(feat)
    corr = float(np.corrcoef(extra['Debt_to_Income_Ratio'], feat['Debt_to_Income'])[0, 1])
    variants = {'bộ chính (80 cột)': X_main,
                'bộ chính + 3 đặc trưng của đề': pd.concat([X_main, extra], axis=1),
                'bộ chính + Monthly_Gross_Income + Monthly_Savings': pd.concat(
                    [X_main, extra[['Monthly_Gross_Income', 'Monthly_Savings']]], axis=1)}
    per_variant, f1_by_variant = {}, {}
    for name, X in variants.items():
        scores = fold_scores(make_model, X.iloc[train_idx].reset_index(drop=True), y, inp['folds5'])
        per_variant[name] = {'số cột': int(X.shape[1]), **summarize(scores)}
        f1_by_variant[name] = scores['macro_f1']
        say(f'[features] {name:52s} macro-F1 {scores.macro_f1.mean():.4f} ± {scores.macro_f1.std(ddof=0):.4f}'
            f'  accuracy {scores.accuracy.mean():.4f}')
        if name.startswith('bộ chính (80'):
            per_class = scores
    base = f1_by_variant['bộ chính (80 cột)']
    verdict = {}
    for name in list(variants)[1:]:
        gain = f1_by_variant[name] - base
        verdict[name] = {'chênh macro-F1 từng phần': [round(float(v), 5) for v in gain],
                         'số phần tốt hơn': int((gain > 0).sum()), 'đưa vào bộ chính': bool((gain > 0).all())}
        say(f'[features] {name}: tốt hơn ở {int((gain > 0).sum())}/5 phần, chênh trung bình {gain.mean():+.4f}')
    return {'tương quan Debt_to_Income_Ratio với Debt_to_Income đã có': corr, 'các bộ đặc trưng': per_variant,
            'kết luận theo quy tắc (cao hơn ở cả 5 phần)': verdict, 'chỉ số theo lớp của bộ chính trong CV': summarize(per_class)}


# ---------------------------------------------------------------------------
# Kiểm tra 5: chia theo thời gian (giống tình huống của test.csv)
# ---------------------------------------------------------------------------
def score_row(y_true, pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=CLASS_CODES, zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, pred)), 'macro_f1': float(f1_score(y_true, pred, average='macro')),
            **{f'recall_{c}': float(v) for c, v in zip(CLASSES, r)}}


def check_timesplit(inp, make_model):
    raw = inp['raw']
    nolabel = cc.add_month_num(raw.drop(columns=cc.TARGET))
    label = raw.set_index('ID')[cc.TARGET]
    early = nolabel[nolabel['Month_Num'] <= 4]

    # phần "huấn luyện" (tháng 1 đến 4) được làm sạch riêng, phần "test" (tháng 5 đến 8) làm sạch cùng lịch sử 1 đến 4,
    # đúng như make_submission.py làm với train và test thật
    clean_early = cs.clean_panel(early)
    X_early = cs.feature_matrix(cf.build_features(clean_early))
    y_early = cs.encode_labels(pd.Series(label.loc[clean_early['ID']].to_numpy()))
    clean_8 = cs.clean_panel(nolabel)
    late = (clean_8['Month_Num'] > 4).to_numpy()
    X_late = cs.align_columns(cs.feature_matrix(cf.build_features(clean_8)[late]), X_early.columns)
    y_late = cs.encode_labels(pd.Series(label.loc[clean_8.loc[late, 'ID']].to_numpy()))
    cust_early, cust_late = clean_early['Customer_ID'].reset_index(drop=True), clean_8.loc[late, 'Customer_ID'].reset_index(drop=True)

    # (i) cùng khách, tháng sau: giống tình huống test.csv
    same = make_model().fit(X_early, y_early)
    same_customers = score_row(y_late, same.predict(X_late))

    # (ii) khách chưa từng thấy, tháng sau: huấn luyện trên 80% khách, chấm trên 20% khách còn lại
    fit_c, new_c = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(X_early, y_early, cust_early))
    new_customer_ids = set(cust_early.iloc[new_c])
    new_late = cust_late.isin(new_customer_ids).to_numpy()
    unseen = make_model().fit(X_early.iloc[fit_c], y_early.iloc[fit_c])
    unseen_customers = score_row(y_late[new_late], unseen.predict(X_late[new_late]))

    majority = float((y_late == y_early.mode().iloc[0]).mean())
    result = {'huấn luyện': 'tháng 1 đến 4 (50.000 dòng)', 'chấm điểm': 'tháng 5 đến 8',
              'cùng khách, tháng sau (giống test.csv)': same_customers,
              'khách chưa từng thấy, tháng sau': unseen_customers,
              'luôn đoán lớp phổ biến của train, accuracy': majority}
    say(f'[timesplit] cùng khách, tháng sau   : accuracy {same_customers["accuracy"]:.4f}, macro-F1 {same_customers["macro_f1"]:.4f}')
    say(f'[timesplit] khách mới, tháng sau    : accuracy {unseen_customers["accuracy"]:.4f}, macro-F1 {unseen_customers["macro_f1"]:.4f}')
    say(f'[timesplit] luôn đoán Standard      : accuracy {majority:.4f}')
    return result


# ---------------------------------------------------------------------------
def main(checks):
    start = time.time()
    inp = load_inputs()
    model_name, make_model = final_model_factory()
    say(f'Mô hình cuối: {model_name}. Tập train của CV: {len(inp["train_idx"]):,} dòng, '
        f'{inp["cust_train"].nunique():,} khách')
    results = {'mô hình': model_name}
    runners = {'folds': lambda: check_folds(inp), 'regression': lambda: check_regression(inp),
               'imputer': lambda: check_imputer(inp, make_model), 'features': lambda: check_features(inp, make_model),
               'timesplit': lambda: check_timesplit(inp, make_model)}
    path = DATA / 'reports' / 'robustness_checks.json'
    if path.exists():                                     # chạy từng phần thì giữ kết quả các phần đã chạy trước
        results.update(json.loads(path.read_text(encoding='utf-8')))
    for name in checks:
        t = time.time()
        results[name] = runners[name]()
        say(f'--- xong "{name}" sau {time.time() - t:.0f}s\n')
        path.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=lambda o: o.item()), encoding='utf-8')
    say(f'Tổng thời gian {time.time() - start:.0f}s. Đã ghi {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checks', default='all', help='danh sách cách nhau bởi dấu phẩy: ' + ', '.join(ALL_CHECKS))
    wanted = parser.parse_args().checks
    selected = ALL_CHECKS if wanted == 'all' else wanted.split(',')
    unknown = set(selected) - set(ALL_CHECKS)
    assert not unknown, f'không có kiểm tra: {unknown}'
    main(selected)
