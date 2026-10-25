"""Các kiểm tra thêm cho pipeline: folds, regression, imputer, features, timesplit.
Kết quả ghi vào data/reports/robustness_checks.json. Chạy từ thư mục gốc:

    python3 check_robustness.py                       # chạy tất cả, khoảng 30 phút
    python3 check_robustness.py --checks folds,timesplit
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
import credit_output as cs          # noqa: E402

DATA = ROOT / 'data'
CLASSES = ['Poor', 'Standard', 'Good']
CLASS_CODES = [cs.LABEL_TO_CODE[c] for c in CLASSES]
ALL_CHECKS = ['folds', 'regression', 'imputer', 'features', 'timesplit']


def load_inputs():
    raw = pd.read_csv(DATA / 'train.csv', low_memory=False)
    clean = pd.read_csv(DATA / 'credit_clean.csv')
    feat = pd.read_csv(DATA / 'credit_features.csv')
    assert (raw['ID'].to_numpy() == clean['ID'].to_numpy()).all() and (clean['ID'].to_numpy() == feat['ID'].to_numpy()).all()
    y_text, customer = clean[cc.TARGET], clean['Customer_ID']

    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(clean, y_text, customer))
    y_train_text = y_text.iloc[train_idx].reset_index(drop=True)
    cust_train = customer.iloc[train_idx].reset_index(drop=True)
    folds5 = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
                  .split(np.zeros(len(y_train_text)), y_train_text, cust_train))
    folds3 = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=7)
                  .split(np.zeros(len(y_train_text)), y_train_text, cust_train))
    return dict(raw=raw, clean=clean, feat=feat, train_idx=train_idx, test_idx=test_idx, y_train_text=y_train_text,
                y_train=cs.encode_labels(y_train_text), cust_train=cust_train, folds5=folds5, folds3=folds3)


def final_model_factory():
    name, params, class_weight = cs.load_chosen_model(DATA / 'reports' / 'final_test_metrics.json')
    return name, (lambda: cs.make_model(name, params, class_weight, seed=0))


def fold_scores(make_model, X, y, folds):
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
    return {col: {'mean': float(scores[col].mean()), 'std': float(scores[col].std(ddof=0))} for col in scores.columns}


def say(text):
    print(text, flush=True)


def check_folds(inp):
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


def check_regression(inp):
    raw = inp['raw']
    expected = cc.clean(raw).drop(columns=cc.TARGET)
    got = cs.clean_panel(cc.add_month_num(raw.drop(columns=cc.TARGET)))
    identical = expected.equals(got)
    differing = 0 if identical else int((expected.ne(got) & ~(expected.isna() & got.isna())).sum().sum())
    say(f'[regression] clean_panel (8 tháng) so với cc.clean(): giống hệt = {identical}, số ô khác = {differing}')
    assert identical, 'clean_panel khác cc.clean() trên cùng dữ liệu 8 tháng'
    return {'giống hệt cc.clean() trên 8 tháng': identical, 'số ô khác': differing}


def prepare_before_money_fill(raw):
    d = cc.add_month_num(raw.drop(columns=cc.TARGET))
    d = cc.mark_placeholders(d)
    d = cc.strip_numeric_text(d)
    d = cc.mark_invalid(d)
    return cc.repair(d, plan={**cc.REPAIR_PLAN, 'model_ratio': []})


# fit_rows: các dòng mô hình được học, None là cách của pipeline chính
def fill_money_columns(d_before, fit_rows=None):
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
    clean_all = fill_money_columns(d_before)
    same_as_pipeline = clean_all.equals(cc.clean(raw).drop(columns=cc.TARGET))
    say(f'[imputer] dựng lại pipeline chính từng bước: giống hệt cc.clean() = {same_as_pipeline}')
    assert same_as_pipeline
    X_all = cs.feature_matrix(cf.build_features(clean_all)).iloc[train_idx].reset_index(drop=True)
    standard = fold_scores(make_model, X_all, y, inp['folds5'])
    say(f'[imputer] học trên mọi dòng      : macro-F1 {standard.macro_f1.mean():.4f} ± {standard.macro_f1.std(ddof=0):.4f}'
        f' ({time.time() - t:.0f}s)')

    refit_rows = []
    for k, (fit_rows, val_rows) in enumerate(inp['folds5'], 1):
        fit_positions = train_idx[fit_rows]
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


def spec_features(feat):
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


def score_row(y_true, pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, pred, labels=CLASS_CODES, zero_division=0)
    return {'accuracy': float(accuracy_score(y_true, pred)), 'macro_f1': float(f1_score(y_true, pred, average='macro')),
            **{f'recall_{c}': float(v) for c, v in zip(CLASSES, r)}}


def check_timesplit(inp, make_model):
    raw = inp['raw']
    nolabel = cc.add_month_num(raw.drop(columns=cc.TARGET))
    label = raw.set_index('ID')[cc.TARGET]
    early = nolabel[nolabel['Month_Num'] <= 4]

    clean_early = cs.clean_panel(early)
    X_early = cs.feature_matrix(cf.build_features(clean_early))
    y_early = cs.encode_labels(pd.Series(label.loc[clean_early['ID']].to_numpy()))
    clean_8 = cs.clean_panel(nolabel)
    late = (clean_8['Month_Num'] > 4).to_numpy()
    X_late = cs.align_columns(cs.feature_matrix(cf.build_features(clean_8)[late]), X_early.columns)
    y_late = cs.encode_labels(pd.Series(label.loc[clean_8.loc[late, 'ID']].to_numpy()))
    cust_early, cust_late = clean_early['Customer_ID'].reset_index(drop=True), clean_8.loc[late, 'Customer_ID'].reset_index(drop=True)

    # (i) cùng khách, tháng sau (giống test.csv)
    same = make_model().fit(X_early, y_early)
    same_customers = score_row(y_late, same.predict(X_late))

    # (ii) khách chưa từng thấy, tháng sau: học trên 80% khách, chấm trên 20% còn lại
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
    if path.exists():
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
