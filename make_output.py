"""Tạo outputs/output.csv cho test.csv. Chạy từ thư mục gốc của dự án (cần data/train.csv và data/test.csv):

    python3 make_output.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import credit_cleaning as cc            # noqa: E402
import credit_features as cf            # noqa: E402
import credit_output as cs          # noqa: E402

SEEDS = {'mô hình cuối (Random Forest random_state)': 0, 'mô hình điền thiếu HistGradientBoosting': 0,
         'chia tập kiểm tra giữ kín (notebook 4)': 42, 'CV 5 phần (notebook 4)': 42, 'CV 3 phần để chỉnh tham số (notebook 4)': 7}


def load_clean_train(train_raw, clean_path):
    if clean_path.exists():
        clean = pd.read_csv(clean_path)
    else:
        print(f'Chưa có {clean_path.name}, làm sạch lại từ train.csv ...')
        clean = cc.clean(train_raw)
        clean.to_csv(clean_path, index=False)
        clean = pd.read_csv(clean_path)
    assert (clean['ID'].to_numpy() == train_raw['ID'].to_numpy()).all(), 'credit_clean.csv lệch thứ tự dòng so với train.csv'
    return clean


def main(args):
    start = time.time()
    data_dir, report_dir = ROOT / 'data', ROOT / 'data' / 'reports'
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = {'seeds': SEEDS}

    # 1. đọc dữ liệu
    train_raw, test_raw = cs.load_raw(args.train, args.test)
    log['cấu trúc dữ liệu'] = cs.describe_structure(train_raw, test_raw)
    print('Train', train_raw.shape, '| Test', test_raw.shape,
          '| khách của test cũng có trong train:', log['cấu trúc dữ liệu']['test_customers_also_in_train'])

    noise = pd.concat({'train': cs.noise_rates(train_raw.drop(columns=cc.TARGET)), 'test': cs.noise_rates(test_raw)}, axis=1)
    noise = noise.swaplevel(axis=1).sort_index(axis=1)
    log['mức nhiễu train so với test (tỷ lệ dòng)'] = {f'{col} | {kind} | {part}': round(float(v), 4)
                                                      for col, row in noise.iterrows() for (kind, part), v in row.items()}
    print('\nTỷ lệ dòng có nhiễu (chưa sửa), train so với test:')
    print((100 * noise).round(2).to_string())

    # 2. làm sạch chuỗi 12 tháng
    panel_raw = cs.build_panel(train_raw, test_raw)
    by_customer = panel_raw.groupby('Customer_ID')['Month_Num']
    assert by_customer.apply(lambda s: s.is_monotonic_increasing).all(), 'thứ tự tháng của một khách không tăng dần'
    breaks = cs.credit_history_breaks(panel_raw)
    assert breaks == 0, f'{breaks} khách có Credit_History_Age không nối tiếp giữa train và test'
    log['khách có Credit_History_Age không tăng đúng 1 tháng mỗi tháng (12 tháng, ô đọc được)'] = breaks
    print(f'\nCredit_History_Age tăng đúng 1 tháng mỗi tháng xuyên 12 tháng ở mọi khách (số khách vi phạm: {breaks})')
    panel_clean, clean_log, marked = cs.clean_panel(panel_raw, return_log=True)
    checks = cs.validate_panel(panel_clean, n_months=12)
    assert all(checks.values()), f'dữ liệu sạch không đạt: {[k for k, v in checks.items() if not v]}'
    is_test = (panel_clean['Month_Num'] > 8).to_numpy()
    test_clean = panel_clean[is_test]
    assert (test_clean['ID'].to_numpy() == test_raw['ID'].to_numpy()).all(), 'dòng test bị đổi thứ tự khi làm sạch'
    test_clean.to_csv(data_dir / 'credit_test_clean.csv', index=False)

    cells_repaired = marked[is_test].isna().sum()
    log['làm sạch test'] = {'số ô được sửa theo cột (chỉ các dòng test)': {k: int(v) for k, v in cells_repaired.items() if v},
                            'giá trị giả đã đổi sang NaN (cả chuỗi 12 tháng)': clean_log['placeholder'],
                            'giá trị vô lý đã đổi sang NaN (cả chuỗi 12 tháng)': clean_log['invalid'],
                            '12 kiểm tra ràng buộc': checks}
    print(f'Làm sạch xong ({len(test_clean):,} dòng test), {sum(checks.values())}/{len(checks)} kiểm tra đạt')

    # 3. đặc trưng
    panel_feat = cf.build_features(panel_clean)
    X_test = cs.feature_matrix(panel_feat[is_test])
    train_clean = load_clean_train(train_raw, data_dir / 'credit_clean.csv')
    X_train = cs.feature_matrix(cf.build_features(train_clean))
    X_test = cs.align_columns(X_test, X_train.columns)
    y_train = cs.encode_labels(train_raw[cc.TARGET])
    log['đặc trưng'] = {'nhóm': cs.MAIN_GROUPS, 'số cột': int(X_train.shape[1]), 'dòng huấn luyện': len(X_train),
                        'dòng cần dự đoán': len(X_test)}
    print('Ma trận đặc trưng: train', X_train.shape, '| test', X_test.shape)

    # 4. huấn luyện lại trên toàn bộ train, dự đoán test
    name, params, class_weight = cs.load_chosen_model(report_dir / 'final_test_metrics.json')
    model = cs.make_model(name, params, class_weight, seed=SEEDS['mô hình cuối (Random Forest random_state)'])
    model.fit(X_train, y_train)
    predicted = cs.decode_labels(model.predict(X_test))
    log['mô hình'] = {'tên': name, 'siêu tham số': params, 'class_weight': class_weight,
                      'chọn từ': 'data/reports/final_test_metrics.json (notebook 4, CV chia theo khách)'}

    # 5. ghi file output và kiểm tra
    cs.build_output(test_raw, predicted).to_csv(out_path, index=False)
    log['kiểm tra file output'] = cs.check_output(out_path, test_raw)
    share = lambda s: {k: round(float(v), 4) for k, v in s.value_counts(normalize=True).items()}
    log['phân bố nhãn dự đoán trên test'] = share(pd.Series(predicted))
    log['phân bố nhãn trong train (để so)'] = share(train_raw[cc.TARGET])
    log['thời gian chạy (giây)'] = round(time.time() - start)
    with open(report_dir / 'output_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=1, default=str)

    print(f'\nĐã ghi {out_path} ({len(predicted):,} dòng). Mô hình: {name}')
    print('Kiểm tra file:', *[f'\n  [{"đạt" if ok else "SAI"}] {k}' for k, ok in log['kiểm tra file output'].items()])
    print('Phân bố nhãn dự đoán:', log['phân bố nhãn dự đoán trên test'])
    print('Phân bố nhãn train  :', log['phân bố nhãn trong train (để so)'])
    print(f'Xong sau {log["thời gian chạy (giây)"]} giây')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tạo output.csv cho test.csv')
    parser.add_argument('--train', default=str(ROOT / 'data' / 'train.csv'))
    parser.add_argument('--test', default=str(ROOT / 'data' / 'test.csv'))
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'output.csv'))
    main(parser.parse_args())
