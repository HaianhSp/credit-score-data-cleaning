#!/bin/bash
# Chạy lần lượt notebook 1, 2, 3, 4 từ đầu đến cuối, rồi tạo file nộp bài. Dừng ngay nếu một bước bị lỗi.
# Cách dùng, từ thư mục gốc của dự án:  bash run_all.sh
# Cần có data/train.csv (tải từ Kaggle) trước khi chạy. Có thêm data/test.csv thì tạo luôn outputs/submission.csv.
# (check_robustness.py mất khoảng 20 đến 30 phút nên không nằm trong script này, chạy riêng nếu cần.)

cd "$(dirname "$0")/notebooks" || exit 1

for nb in 1_data_understanding_eda.ipynb 2_data_cleaning.ipynb 3_post_cleaning_features.ipynb 4_modeling_evaluation.ipynb; do
  python3 ../run.py "$nb" 2>&1 | grep -v "IPKernelApp"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "Dừng: $nb bị lỗi"
    exit 1
  fi
done
echo "Đã chạy xong cả 4 notebook"

cd .. || exit 1
if [ -f data/test.csv ]; then
  python3 make_submission.py || { echo "Dừng: make_submission.py bị lỗi"; exit 1; }
else
  echo "Không thấy data/test.csv, bỏ qua bước tạo outputs/submission.csv"
fi
