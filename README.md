# Phân tích, làm sạch dữ liệu và phân loại điểm tín dụng khách hàng

**Môn:** Khai phá dữ liệu  |  **Bộ dữ liệu:** [Credit Score Classification (Kaggle)](https://www.kaggle.com/datasets/parisrohan/credit-score-classification) – `train.csv`
**Sinh viên:** Lê Nguyễn Hải Anh – Đại học Phenikaa

## 1. Đề tài làm gì và vì sao

Dự đoán **điểm tín dụng** (`Credit_Score` ∈ {Poor, Standard, Good}) của khách hàng ngân hàng từ hồ sơ tài chính hằng tháng.

Bộ dữ liệu này được chọn **vì nó rất bẩn**: 100.000 dòng (= 12.500 khách × 8 tháng) với giá trị giả (`_______`, `#F%$D@*&8`, `__10000__`, `-333…333`), số liệu viết dính dấu `_`, giá trị vô lý (tuổi −500 và 8.698, 1.499 thẻ tín dụng, lãi suất 5.797 %, EMI 82.331…), thiếu dữ liệu và không nhất quán giữa các tháng của cùng một khách. Vì vậy **phân tích và làm sạch dữ liệu là phần chính của bài**; mô hình học máy chỉ là phần cuối, dùng để *đo* làm sạch có giá trị bao nhiêu.

Quy trình theo KDD / CRISP-DM, mỗi bước là một notebook, mỗi quyết định đều có bằng chứng bằng số liệu:

```
train.csv (bẩn) ─▶ [1] Hiểu dữ liệu & EDA ─▶ [2] Làm sạch ─▶ credit_clean.csv ─▶ [3] Kiểm định + tạo đặc trưng ─▶ credit_features.csv ─▶ [4] Mô hình & đánh giá
```

## 2. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `notebooks/1_data_understanding_eda.ipynb` | Từ điển dữ liệu, cấu trúc khách × tháng, đo từng loại lỗi bằng nhiều cách độc lập, bảng *phát hiện → bằng chứng → quyết định* |
| `notebooks/2_data_cleaning.ipynb` | Làm sạch 6 bước; **chọn chiến lược điền thiếu bằng thí nghiệm che giá trị**; 12 ràng buộc kiểm tra; báo cáo chất lượng trước/sau; xuất `credit_clean.csv` |
| `notebooks/3_post_cleaning_features.ipynb` | Kiểm tra làm sạch có làm méo dữ liệu / tạo tín hiệu giả không; tạo và đo giá trị của 5 nhóm đặc trưng; xuất `credit_features.csv` |
| `notebooks/4_modeling_evaluation.ipynb` | Thí nghiệm *ablation* theo từng bước làm sạch, so sánh mô hình, tối ưu siêu tham số, kết quả cuối trên tập giữ kín, độ quan trọng đặc trưng, phân tích lỗi |
| `src/credit_cleaning.py` | **Nguồn duy nhất** của pipeline làm sạch `clean()` (notebook 2 trích nguyên văn từ file này và kiểm tra kết quả trùng khớp) |
| `src/credit_features.py` | Tạo đặc trưng |
| `data/train.csv` | Dữ liệu gốc (tải từ Kaggle; không đưa lên kho mã) |
| `data/credit_clean.csv` | **Dữ liệu sau làm sạch** (100.000 × 29, không còn ô thiếu / giá trị giả / vô lý) |
| `data/credit_features.csv` | Dữ liệu sạch + 45 đặc trưng tự tạo (sinh lại bằng notebook 3) |
| `data/reports/` | Báo cáo chất lượng dữ liệu trước/sau, nhật ký làm sạch, các bảng kết quả mô hình |
| `run_all.sh`, `run.py` | Chạy lần lượt cả 4 notebook từ đầu đến cuối |

**Chạy lại:** đặt `train.csv` vào `data/`, cài `pandas numpy scikit-learn scipy matplotlib seaborn nbformat nbclient ipykernel`, rồi `bash run_all.sh` (≈ 1–1,5 giờ, chủ yếu là notebook 4). Hoặc mở từng notebook và *Restart & Run All*.


## 3. Phân tích dữ liệu → quyết định làm sạch

Dữ liệu **không phải 100.000 quan sát độc lập** mà là 12.500 khách × 8 tháng; nhãn đổi chậm theo tháng, lệch lớp (Standard 53 %, Poor 29 %, Good 18 %). Hệ quả: (1) các tháng khác của *chính khách đó* là nguồn tốt nhất để sửa ô lỗi; (2) phải chia train/test **theo khách** để không rò rỉ.

Mức độ bẩn (notebook 1, mục 1.9): **135.774 / 2.800.000 ô (4,8 %) có vấn đề; 76,3 % số dòng có ít nhất một ô lỗi; 100 % khách hàng có lỗi** → xóa dòng lỗi không khả thi (sau bước 1–3 xóa dòng còn lại 29,5 % dữ liệu và **không còn khách nào đủ 8 tháng**), bắt buộc phải *sửa / khôi phục*.

Bảng *phát hiện → bằng chứng → quyết định* (chi tiết và biểu đồ ở notebook 1, mục 1.10):

| # | Vấn đề | Bằng chứng | Quyết định |
|---|---|---|---|
| 1 | 7 cột có giá trị giả (`#F%$D@*&8`, `_______`, `_`, `!@9#%8`, `__10000__`, `__-333…333__`) | 3 cách phát hiện độc lập: không chứa chữ cái / sai định dạng / lặp bất thường & độ lớn bất khả; nhóm placeholder không mang thông tin nhãn | Đổi → NaN **trước** khi ép kiểu (bỏ `_` của `__10000__` sẽ thành số 10000 "trông như thật") |
| 2 | 8 cột số lưu thành chữ dính `_` (`'24_'`) | Bỏ `_` đầu/cuối → đọc được 100 % ô còn lại | Bỏ `_` rồi ép `float64` |
| 3 | Cột đếm có giá trị ngoài miền (tuổi −500, 8.698; 1.499 thẻ…) | "Vách đá" tần suất: mỗi giá trị trong miền thật xuất hiện ≥ 50 lần, ngoài miền chỉ vài lần; kết quả **không nhạy** với ngưỡng 10–100 | Ngoài miền → NaN |
| 4 | EMI rác | EMI thật có phần lẻ dài và ≤ 1.779; 3.164 giá trị là số nguyên dương tới 82.331 | EMI > 1800 hoặc là số nguyên dương → NaN |
| 5 | `Annual_Income` rác | Thu nhập thật không đổi trong 8 tháng; giá trị rác lệch hàng bậc độ lớn so với chính khách | Lệch > 50 % so với trung vị khách → NaN |
| 6 | **Không** dùng IQR chung | IQR xóa nhầm hàng nghìn giá trị thật ở 7 cột | Chỉ áp luật cho cột có bằng chứng rác |
| 7 | Cột hằng số theo khách (`SSN`, `Occupation`, `Name`, `Credit_Mix`, `Annual_Income`, `Interest_Rate`, `Outstanding_Debt`) | 1 giá trị / khách ở 12.500 khách | Điền bằng mode / median **của khách** |
| 8 | Cột có "bước nhảy" thật (số tài khoản, số thẻ, lương, EMI, tuổi, truy vấn tín dụng) | Thay đổi đi theo cặp và một lần (đổi việc, thêm khoản vay…) | Điền bằng **tháng gần nhất** (không dùng mode: sẽ kéo giai đoạn này sang giai đoạn kia) |
| 9 | `Num_of_Loan` rác | Khớp 100 % số loại vay trong `Type_of_Loan` (trên 95.652 dòng hợp lệ) | Suy ra từ `Type_of_Loan` |
| 10 | `Credit_History_Age` là chuỗi, có thiếu | Tăng đúng +1 tháng/tháng ở 100 % khách | Đổi ra số tháng; suy ô thiếu bằng số học (khôi phục đúng 100 % ô bị che) |
| 11 | `Type_of_Loan` thiếu 11,4 % | Thiếu ⇔ `Num_of_Loan = 0` | Điền `No Loan` (không phải dữ liệu mất) |
| 12 | `Amount_invested_monthly`, `Monthly_Balance` biến động mạnh | Phụ thuộc lương và hành vi chi tiêu của cùng dòng; cách điền theo khách không hơn đường cơ sở | Mô hình học máy dự đoán **tỷ lệ so với lương** (thắng dự đoán giá trị tuyệt đối) |
| 13 | `Payment_Behaviour` ngẫu nhiên theo tháng | Không chiến lược nào đoán đúng hơn ngẫu nhiên (≈ 0,23–0,28) | Giữ thành nhóm `Unknown` thay vì bịa giá trị |
| 14 | Nhãn lệch lớp; dữ liệu theo khách | Mục 1.2 | Chia train/test theo khách; chỉ số chính là macro-F1 |

**Không dùng `Credit_Score` ở bất kỳ bước làm sạch nào** (tránh rò rỉ nhãn vào đặc trưng).

## 4. Làm sạch: kết quả và cách kiểm chứng

**Chiến lược điền thiếu được chọn bằng thí nghiệm, không đoán** (notebook 2, mục 4.2): lấy các ô đã biết hợp lệ, che ngẫu nhiên 10 %, điền lại bằng từng chiến lược, đo sai số chuẩn hóa NMAE = MAE/σ (0 = hoàn hảo, ≈ 0,8 = đoán bừa bằng trung vị), lặp 3 hạt ngẫu nhiên. Bảng kết quả chọn chiến lược tốt nhất cho từng cột, và notebook **dừng (assert) nếu kế hoạch trong module không khớp thí nghiệm**.

| Nhóm cột | Chiến lược thắng | Sai số (NMAE) so với điền trung vị toàn cục |
|---|---|---|
| `Annual_Income`, `Interest_Rate`, `Outstanding_Debt` | median của khách | 0,000 so với 0,74–0,83 (khôi phục chính xác) |
| `Age`, số tài khoản, số thẻ, EMI, lương, truy vấn tín dụng | tháng gần nhất | 0,001–0,067 so với 0,61–0,84 |
| `Num_of_Delayed_Payment`, `Changed_Credit_Limit` | median của khách | 0,12 / 0,11 so với 0,83 / 0,78 |
| `Amount_invested_monthly`, `Monthly_Balance` | mô hình dự đoán tỷ lệ so với lương | 0,052 / 0,075 (dự đoán giá trị tuyệt đối: 0,077 / 0,090; theo khách: 0,57 / 0,44) |
| `SSN`, `Occupation`, `Name`, `Credit_Mix` | mode của khách | độ chính xác 100 % (điền mode toàn cục: 0–46 %) |

**Trước → sau** (`data/reports/data_quality_report.csv`): 126.926 ô thiếu / giả / vô lý (4,53 % số ô) → **0**; 12 / 12 ràng buộc kiểm tra đều đạt (không còn ô thiếu, mọi cột trong miền hợp lệ, mỗi khách đúng 1 SSN/nghề/tên/`Credit_Mix`, `Num_of_Loan` = số loại vay, lịch sử tín dụng tăng +1/tháng…). Kết quả của notebook và của module `clean()` **giống hệt từng ô**; chạy lại cho file CSV giống hệt (so sánh byte).

**Kiểm tra làm sạch có làm hỏng dữ liệu không** (notebook 3): phân phối ô được điền bám sát ô gốc ở 11/12 cột (KS ≤ 0,08, σ không bị co); quan hệ với nhãn giữ nguyên (thứ tự mức liên quan không đổi, `Credit_Mix` mạnh lên rõ vì nhóm placeholder 20 % dòng được trả về giá trị thật); cấu trúc tương quan giữa 16 cột số gần như không đổi (chênh trung bình 0,002).

**Hạn chế cần nêu rõ – `Monthly_Balance`:** 1.209 ô thiếu (1,2 %) **không phải mẫu ngẫu nhiên** (lương và đầu tư cao hơn nhiều; 73 % là `Low_spent_Small_value` so với 25 % ở các dòng còn lại). Mô hình điền chính xác trên các ô đã biết (kể cả vùng lương cao), nhưng với các ô thiếu thật nó dự đoán số dư ≤ 0 cho hơn một nửa, trong khi dữ liệu gốc không có số dư nào ≤ 0 – dấu hiệu phù hợp với *thiếu không ngẫu nhiên* (số dư bị bỏ trống vì ≤ 0). Đây là **giả thuyết, không kiểm chứng được**; ảnh hưởng giới hạn ở ~1,2 % số dòng của một cột (xem phân tích lỗi ở notebook 4).



## 5. Mô hình: làm sạch đóng góp bao nhiêu?

Mô hình đến **sau cùng** và chỉ để *đo* giá trị của làm sạch (notebook 4).

**Thiết kế đánh giá (để con số đáng tin):**
- Mỗi khách có 8 dòng gần giống nhau, nên chia ngẫu nhiên theo dòng sẽ để cùng một người ở cả train lẫn test (rò rỉ). Bài chia **theo khách hàng**: 20 % khách (2.500 người, đủ 8 tháng) làm tập kiểm tra giữ kín, 80 % còn lại dùng cho mọi lựa chọn bằng CV 5 phần theo khách (`StratifiedGroupKFold`).
- Nhãn lệch lớp (Standard 53 %) → chỉ số chính là **macro-F1**; tập kiểm tra chỉ chấm **một lần** ở cuối; khoảng tin cậy 95 % bằng bootstrap theo khách.

**Thí nghiệm chính – ablation theo mức làm sạch** (cùng 21 cột gốc, cùng mô hình, cùng các phần CV; chỉ khác dữ liệu):

| Giai đoạn | Mô tả | Logistic Regression | Random Forest | HistGradientBoosting |
|---|---|---|---|---|
| S0 – Thô | chỉ bỏ `_` và ép số; placeholder còn nguyên (kiểu "tải về là chạy") | 0,589 | 0,669 | 0,662 |
| S1 | S0 + placeholder → NaN, điền trung vị/mode toàn cục | 0,585 | 0,667 | 0,660 |
| S2 | S1 + bỏ giá trị vô lý, điền toàn cục | 0,612 | 0,667 | 0,657 |
| S2* | S2, để mô hình tự xử lý NaN (không điền) | – | – | 0,663 |
| **S3 – Làm sạch đầy đủ** | pipeline của bài (= `credit_clean.csv`) | **0,639** | **0,679** | **0,671** |

(macro-F1, trung bình CV 5 phần; độ lệch chuẩn 0,001–0,007, xem `data/reports/ablation_cleaning.csv`.)

- S3 → hơn S0 ở **mọi phần CV** với cả ba mô hình: **+5,1 điểm %** (Logistic Regression), **+1,0** (Random Forest), **+0,9** (HistGradientBoosting). Trên tập kiểm tra giữ kín: +5,7 và +0,5 điểm % (Logistic Regression / HistGradientBoosting).
- Mô hình tuyến tính hưởng lợi nhiều nhất vì bị phá bởi ngoại lệ cực đoan (một giá trị −3·10²⁶ làm phép chuẩn hóa vô nghĩa). Mô hình cây chịu ngoại lệ nên lợi ích nhỏ hơn – **đây là mức đóng góp thật, khiêm tốn, không phải điều kỳ diệu**.
- **Làm sạch "máy móc" không đủ:** với mô hình cây, chỉ bỏ placeholder/giá trị vô lý rồi điền trung vị toàn cục (S1, S2) *không hơn dữ liệu thô* (trong sai số), và để mô hình tự xử lý NaN (S2*) cũng không giúp (+0,1 điểm). Lợi ích chỉ xuất hiện khi điền **theo từng khách và theo ràng buộc giữa các cột** (S3 hơn S2* +0,8 điểm %).

**Đặc trưng tự tạo chỉ thứ yếu** (5 nhóm, notebook 3 + mục 4.3): thêm từng nhóm làm macro-F1 đổi trong khoảng −0,2…+0,4 điểm %, nằm trong sai số của CV. Nhóm duy nhất nhìn thấy được là đặc trưng theo thời gian (`LAG`) cho mô hình tuyến tính (+0,4 điểm). Đặc trưng "nhìn lại 8 tháng" (dùng cả các tháng sau) chỉ hơn bản nhân quả vài phần mười điểm % → không dùng làm kết quả chính.

**Mô hình cuối** – Random Forest (tối ưu bằng tìm kiếm ngẫu nhiên, CV 3 phần theo khách; `max_depth=12, max_features=0,3, min_samples_leaf=9, n_estimators=159`), dữ liệu sạch + đặc trưng nhân quả, trên **2.500 khách chưa từng dùng để chọn gì**:

| | accuracy | macro-F1 |
|---|---|---|
| Luôn đoán Standard (đường cơ sở) | 0,529 | 0,231 |
| Logistic Regression – dữ liệu thô | 0,617 | 0,587 |
| HistGradientBoosting – dữ liệu thô | 0,700 | 0,678 |
| HistGradientBoosting – dữ liệu sạch | 0,705 | 0,683 |
| **Random Forest (tuned) – dữ liệu sạch + đặc trưng nhân quả** | **0,714** (CI 95 %: 0,702–0,726) | **0,700** (CI 95 %: 0,685–0,711) |

ROC-AUC (OvR, macro) = 0,861. Trong 2,2 điểm % macro-F1 mà mô hình cuối hơn HistGradientBoosting mặc định trên dữ liệu thô, **0,5 điểm là nhờ làm sạch** và 1,6 điểm nhờ chọn mô hình / tối ưu siêu tham số.

**Mô hình dựa vào gì và sai ở đâu** (mục 4.8–4.9):
- Quan trọng nhất (permutation importance): `Interest_Rate`, `Credit_Mix`, `Outstanding_Debt`, `Num_Credit_Card`, độ trễ thanh toán trung bình lũy kế – toàn là chỉ số nợ / thanh toán, hợp lý về tài chính và khớp với EDA; không dựa vào định danh hay tháng.
- Sai nhiều nhất ở **các tháng nhãn vừa đổi so với tháng trước** (accuracy 0,42 so với 0,76 khi nhãn giữ nguyên; chiếm 32 % số lỗi dù chỉ ~18 % số dòng có tháng trước). Nhầm trực tiếp Poor ↔ Good chỉ chiếm 8,7 % số lỗi; phần còn lại là ranh giới với Standard.
- Theo tháng: accuracy ≈ 0,79 ở tháng 1–3 nhưng ≈ 0,67 ở tháng 4–8; tỷ lệ nhãn đổi nhảy lên ~26–27 % ở tháng 4 và 7 (so với 13–15 %) nên giải thích được một phần, phần còn lại chưa xác định (có thể là đặc điểm của dữ liệu tổng hợp). Theo `Credit_Mix`: 0,67 (Bad) / 0,83 (Standard) / 0,57 (Good) – nhóm Good khó nhất.
- Dòng phải sửa càng nhiều ô thì accuracy giảm nhẹ (0,720 → 0,699 từ 0 tới ≥ 3 ô), nhưng mức giảm chưa đạt ý nghĩa thống kê; với 228 dòng có `Monthly_Balance` phải điền không kết luận được gì.

## 6. Hạn chế (nêu thẳng)

1. **Dữ liệu tổng hợp.** Nhãn có nhiều nhiễu theo tháng nên mọi mô hình dừng quanh 0,70–0,71 accuracy kể cả khi biết trước cả 8 tháng. Điều đáng tin ở bài này là **chênh lệch giữa các mức làm sạch trong cùng điều kiện**, không phải con số tuyệt đối; không suy ra hiệu năng trên dữ liệu ngân hàng thật.
2. **Thí nghiệm "che 10 %" giả định ô thiếu thật xuất hiện ngẫu nhiên.** Với `Monthly_Balance` điều này sai: các ô thiếu thuộc nhóm khách khác, và có dấu hiệu *thiếu không ngẫu nhiên* (giả thuyết, không kiểm chứng được). Giá trị điền cho `Amount_invested_monthly` và `Monthly_Balance` là **ước lượng**, không phải số thật; chọn "chặn dưới ở 0" là giả định ít nhất có thể.
3. **Miền hợp lệ suy ra từ tần suất** nên có thể loại nhầm vài giá trị thật hiếm sát biên (kết quả không nhạy với ngưỡng 10–100, nhưng không tuyệt đối).
4. **`Name` còn nhiễu chữ cái** (khoảng 100 tên có ký tự thừa) – không sửa vì không dùng làm đặc trưng.
5. **Các tham số làm sạch được tính trên cả 100.000 dòng** (không dùng nhãn) trước khi chia train/test: không rò rỉ nhãn, nhưng nếu triển khai thật nên học chỉ từ dữ liệu huấn luyện.
6. **Một lần chia train/test**, tìm kiếm siêu tham số giới hạn (20 và 8 cấu hình); chưa dùng `test.csv` (không có nhãn) – có thể áp dụng `clean()` + mô hình cuối để dự đoán.

## 7. Chạy lại

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn nbformat nbclient ipykernel
# đặt train.csv (Kaggle) vào data/
bash run_all.sh          # 1 → 2 → 3 → 4, ≈ 1–1,5 giờ trên 2 CPU, chủ yếu là notebook 4 (≈ 55 phút)
```

Kết quả tái lập được: chạy lại notebook 2 cho `credit_clean.csv` giống hệt từng byte; các hạt ngẫu nhiên đều cố định.

Mã làm sạch dùng lại ở dạng module:
```python
import sys; sys.path.insert(0, 'src')
import pandas as pd, credit_cleaning as cc
clean_df = cc.clean(pd.read_csv('data/train.csv', low_memory=False))
assert all(cc.validate_clean(clean_df).values())
```
