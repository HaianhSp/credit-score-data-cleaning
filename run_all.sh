#!/bin/bash
# Chạy notebook 1 đến 4 rồi tạo file output (cần data/train.csv, có thêm data/test.csv thì tạo luôn file output).
# Chạy từ thư mục gốc: bash run_all.sh
# check_robustness.py mất 20 đến 30 phút nên không nằm trong này.

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
  python3 make_output.py || { echo "Dừng: make_output.py bị lỗi"; exit 1; }
else
  echo "Không thấy data/test.csv, bỏ qua bước tạo outputs/output.csv"
fi
