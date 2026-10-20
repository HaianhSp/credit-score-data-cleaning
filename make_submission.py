"""
Tạo file nộp bài submission.csv (cột ID và Credit_Score) cho test.csv.

Cách chạy, từ thư mục gốc của dự án (cần data/train.csv, data/test.csv; mất khoảng 2 đến 3 phút):
    python3 make_submission.py

Các bước:
  1. Đọc train.csv, test.csv và kiểm tra cấu trúc (test là 4 tháng sau của cùng 12.500 khách).
  2. Nối hai file thành chuỗi 12 tháng của mỗi khách (bỏ nhãn trước khi nối) rồi làm sạch bằng đúng pipeline
     của notebook 2. Kiểm tra 12 ràng buộc trên kết quả.
  3. Tạo đặc trưng cho các dòng của test. Đặc trưng theo thời gian (LAG) chỉ nhìn tháng hiện tại và các tháng trước.
  4. Lấy mô hình và siêu tham số đã chọn ở notebook 4 (chọn bằng CV chia theo khách), huấn luyện lại trên TOÀN BỘ train
     có nhãn (data/credit_clean.csv, kết quả làm sạch của notebook 2), rồi dự đoán test.
  5. Ghi outputs/submission.csv, đọc lại từ đĩa để kiểm tra, ghi nhật ký chạy vào data/reports/submission_log.json.

Nhãn Credit_Score của train chỉ xuất hiện ở bước huấn luyện. Test không có nhãn nên không có điểm số nào để tính ở đây.
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
import credit_submission as cs          # noqa: E402

# Các seed đã dùng trong cả dự án (ghi vào nhật ký để chạy lại được)
SEEDS = {'mô hình cuối (Random Forest random_state)': 0, 'mô hình điền thiếu HistGradientBoosting': 0,
         'chia tập kiểm tra giữ kín (notebook 4)': 42, 'CV 5 phần (notebook 4)': 42, 'CV 3 phần để chỉnh tham số (notebook 4)': 7}


def load_clean_train(train_raw, clean_path):
    """Dữ liệu train đã làm sạch (kết quả notebook 2). Nếu chưa có file thì làm sạch lại từ train.csv."""
    if clean_path.exists():
        clean = pd.read_csv(clean_path)
    else:
        print(f'Chưa có {clean_path.name}, làm sạch lại từ train.csv ...')
        clean = cc.clean(train_raw)
        clean.to_csv(clean_path, index=False)
        clean = pd.read_csv(clean_path)          # đọc lại cho giống đường đi của notebook 3 và 4
    assert (clean['ID'].to_numpy() == train_raw['ID'].to_numpy()).all(), 'credit_clean.csv lệch thứ tự dòng so với train.csv'
    return clean


def main(args):
    start = time.time()
    data_dir, report_dir = ROOT / 'data', ROOT / 'data' / 'reports'
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = {'seeds': SEEDS}

    # ---- 1. Đọc dữ liệu --------------------------------------------------------------------
    train_raw, test_raw = cs.load_raw(args.train, args.test)
    log['cấu trúc dữ liệu'] = cs.describe_structure(train_raw, test_raw)
    print('Train', train_raw.shape, '| Test', test_raw.shape,
          '| khách của test cũng có trong train:', log['cấu trúc dữ liệu']['test_customers_also_in_train'])

    # ---- 2. Làm sạch chuỗi 12 tháng ---------------------------------------------------------
    panel_raw = cs.build_panel(train_raw, test_raw)
    by_customer = panel_raw.groupby('Customer_ID')['Month_Num']
    assert by_customer.apply(lambda s: s.is_monotonic_increasing).all(), 'thứ tự tháng của một khách không tăng dần'
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

    # ---- 3. Đặc trưng -----------------------------------------------------------------------
    panel_feat = cf.build_features(panel_clean)
    X_test = cs.feature_matrix(panel_feat[is_test])
    train_clean = load_clean_train(train_raw, data_dir / 'credit_clean.csv')
    X_train = cs.feature_matrix(cf.build_features(train_clean))
    X_test = cs.align_columns(X_test, X_train.columns)
    y_train = cs.encode_labels(train_raw[cc.TARGET])
    log['đặc trưng'] = {'nhóm': cs.MAIN_GROUPS, 'số cột': int(X_train.shape[1]), 'dòng huấn luyện': len(X_train),
                        'dòng cần dự đoán': len(X_test)}
    print('Ma trận đặc trưng: train', X_train.shape, '| test', X_test.shape)

    # ---- 4. Huấn luyện lại trên toàn bộ train rồi dự đoán ----------------------------------
    name, params, class_weight = cs.load_chosen_model(report_dir / 'final_test_metrics.json')
    model = cs.make_model(name, params, class_weight, seed=SEEDS['mô hình cuối (Random Forest random_state)'])
    model.fit(X_train, y_train)
    predicted = cs.decode_labels(model.predict(X_test))
    log['mô hình'] = {'tên': name, 'siêu tham số': params, 'class_weight': class_weight,
                      'chọn từ': 'data/reports/final_test_metrics.json (notebook 4, CV chia theo khách)'}

    # ---- 5. Ghi và kiểm tra file nộp bài ----------------------------------------------------
    cs.build_submission(test_raw, predicted).to_csv(out_path, index=False)
    log['kiểm tra file nộp bài'] = cs.check_submission(out_path, test_raw)
    share = lambda s: {k: round(float(v), 4) for k, v in s.value_counts(normalize=True).items()}
    log['phân bố nhãn dự đoán trên test'] = share(pd.Series(predicted))
    log['phân bố nhãn trong train (để so)'] = share(train_raw[cc.TARGET])
    log['thời gian chạy (giây)'] = round(time.time() - start)
    with open(report_dir / 'submission_log.json', 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=1, default=str)

    print(f'\nĐã ghi {out_path} ({len(predicted):,} dòng). Mô hình: {name}')
    print('Kiểm tra file:', *[f'\n  [{"đạt" if ok else "SAI"}] {k}' for k, ok in log['kiểm tra file nộp bài'].items()])
    print('Phân bố nhãn dự đoán:', log['phân bố nhãn dự đoán trên test'])
    print('Phân bố nhãn train  :', log['phân bố nhãn trong train (để so)'])
    print(f'Xong sau {log["thời gian chạy (giây)"]} giây')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tạo submission.csv cho test.csv')
    parser.add_argument('--train', default=str(ROOT / 'data' / 'train.csv'))
    parser.add_argument('--test', default=str(ROOT / 'data' / 'test.csv'))
    parser.add_argument('--out', default=str(ROOT / 'outputs' / 'submission.csv'))
    main(parser.parse_args())
