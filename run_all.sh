#!/bin/bash
# Chạy tuần tự notebook 1 → 2 → 3 → 4 (dừng nếu có lỗi). Dùng: bash run_all.sh  (từ thư mục gốc dự án)
# Mỗi notebook được thực thi từ đầu đến cuối ("Restart & Run All") và lưu kèm kết quả.
cd "$(dirname "$0")/notebooks" || exit 1
for nb in 1_data_understanding_eda.ipynb 2_data_cleaning.ipynb 3_post_cleaning_features.ipynb 4_modeling_evaluation.ipynb; do
  python3 ../run.py "$nb" 2>&1 | grep -v "IPKernelApp"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "DỪNG: $nb lỗi"; exit 1; }
done
echo "XONG tất cả"
