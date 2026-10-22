# Phân tích, làm sạch dữ liệu và phân loại điểm tín dụng khách hàng

Môn: Khai phá dữ liệu. Bộ dữ liệu: [Credit Score Classification (Kaggle)](https://www.kaggle.com/datasets/parisrohan/credit-score-classification), hai file `train.csv` (có nhãn) và `test.csv` (không nhãn).
Sinh viên: Lê Nguyễn Hải Anh, Đại học Phenikaa.

## 1. Đề tài

Bài toán là dự đoán điểm tín dụng `Credit_Score` (Poor, Standard hoặc Good) của khách hàng ngân hàng từ hồ sơ tài chính hằng tháng.

Bộ dữ liệu có 100.000 dòng (12.500 khách, mỗi khách 8 tháng) và rất bẩn. Có giá trị giả (`_______`, `#F%$D@*&8`, `__10000__`, `-333…333`), số bị dính dấu `_`, giá trị vô lý (tuổi −500 và 8.698, 1.499 thẻ tín dụng, lãi suất 5.797%, EMI 82.331), ô thiếu, và các tháng của cùng một khách không nhất quán với nhau. Vì vậy phần chính của bài là phân tích và làm sạch dữ liệu. Mô hình học máy đặt ở cuối, chỉ dùng để đo việc làm sạch giúp được bao nhiêu.

Mỗi bước là một notebook dựa trên số liệu:

```
train.csv (dữ liệu chưa qua làm sạch)
   notebook 1: Hiểu dữ liệu, phân tích dữ liệu (EDA)
   notebook 2: Làm sạch dữ liệu                       ->  credit_clean.csv
   notebook 3: Kiểm tra dữ liệu sạch, tạo đặc trưng  ->  credit_features.csv
   notebook 4: Xây dựng mô hình và đánh giá

test.csv (không nhãn)
   make_submission.py: làm sạch, tạo đặc trưng, dự đoán  ->  outputs/submission.csv
```

## 2. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `notebooks/1_data_understanding_eda.ipynb` | Từ điển dữ liệu, cấu trúc khách nhân tháng, đo từng loại lỗi bằng nhiều cách độc lập, bảng phát hiện, bằng chứng và quyết định |
| `notebooks/2_data_cleaning.ipynb` | Làm sạch 6 bước. Cách điền ô thiếu được chọn bằng thí nghiệm che giá trị. Có 12 ràng buộc kiểm tra, báo cáo chất lượng trước và sau, xuất `credit_clean.csv` |
| `notebooks/3_post_cleaning_features.ipynb` | Kiểm tra việc làm sạch có làm méo dữ liệu hoặc tạo tín hiệu giả không. Tạo 5 nhóm đặc trưng và đo giá trị của chúng. Xuất `credit_features.csv` |
| `notebooks/4_modeling_evaluation.ipynb` | Thí nghiệm ablation theo từng mức làm sạch, so sánh mô hình, tối ưu tham số, kết quả cuối trên tập giữ kín, độ quan trọng đặc trưng, phân tích lỗi |
| `src/credit_cleaning.py` | Code làm sạch, gồm hàm `clean()` chạy cả quy trình. Notebook 2 import và in code từ file này ra, nên code trong notebook luôn là code đang chạy |
| `src/credit_features.py` | Code tạo đặc trưng |
| `src/notebook_utils.py` | Hàm dùng chung cho các notebook (in code, màu biểu đồ, vài hàm thống kê) |
| `src/credit_submission.py` | Hàm cho bước dự đoán test: làm sạch chuỗi 12 tháng, dựng ma trận đặc trưng, kiểm tra file nộp bài |
| `make_submission.py` | Script tạo `outputs/submission.csv` cho `test.csv` (mục 6) |
| `check_robustness.py` | Các kiểm tra bổ sung: cách chia CV, học lại mô hình điền thiếu theo từng phần CV, ba đặc trưng thêm, chia theo thời gian (mục 6) |
| `data/train.csv` | Dữ liệu gốc tải từ Kaggle, không đưa lên kho mã |
| `data/test.csv` | File test tải từ Kaggle, không đưa lên kho mã. 50.000 dòng, là 4 tháng sau (September đến December) của chính 12.500 khách trong train |
| `data/credit_clean.csv` | Dữ liệu sau làm sạch (100.000 dòng, 29 cột), không còn ô thiếu, giá trị giả hay giá trị vô lý |
| `data/credit_features.csv` | Dữ liệu sạch cộng 45 đặc trưng tự tạo, sinh lại bằng notebook 3 |
| `data/credit_test_clean.csv` | 50.000 dòng của `test.csv` sau làm sạch |
| `outputs/submission.csv` | File nộp bài: hai cột `ID`, `Credit_Score`, đủ 50.000 dòng theo thứ tự của `test.csv` |
| `data/reports/` | Báo cáo chất lượng dữ liệu trước và sau, nhật ký làm sạch, các bảng kết quả mô hình, `submission_log.json` (cấu hình, seed, kiểm tra file output) và `robustness_checks.json` |
| `run_all.sh`, `run.py` | Chạy lần lượt cả 4 notebook từ đầu đến cuối, rồi tạo file output nếu có `test.csv` |

Cách chạy lại nằm ở mục 8.

## 3. Phân tích dữ liệu và các làm sạch

Dữ liệu không phải 100.000 quan sát độc lập mà là 12.500 khách, mỗi khách 8 tháng. Nhãn đổi chậm theo tháng và lệch lớp (Standard 53%, Poor 29%, Good 18%). Có hai hệ quả:

1. Các tháng khác của chính khách đó là nguồn tốt nhất để sửa một ô lỗi.
2. Phải chia train và test theo khách, nếu không cùng một người sẽ xuất hiện ở cả hai tập.

Mức độ bẩn (notebook 1, mục 1.9): 135.774 trên 2.800.000 ô (4,8%) có vấn đề, 76,3% số dòng có ít nhất một ô lỗi, và mọi khách hàng đều có lỗi. Xóa dòng lỗi không khả thi: sau bước 1 đến 3, xóa dòng chỉ còn lại 29,5% dữ liệu và không còn khách nào đủ 8 tháng. Vì vậy phải sửa hoặc khôi phục.

Bảng dưới đây tóm tắt phát hiện, bằng chứng và quyết định (chi tiết và biểu đồ ở notebook 1, mục 1.10):

| # | Vấn đề | Bằng chứng | Quyết định |
|---|---|---|---|
| 1 | 7 cột có giá trị giả (`#F%$D@*&8`, `_______`, `_`, `!@9#%8`, `__10000__`, `__-333…333__`) | Ba cách phát hiện độc lập: không có chữ cái, sai định dạng, lặp bất thường hoặc độ lớn không thể có. Nhóm giá trị giả không mang thông tin về nhãn | Đổi thành NaN trước khi ép kiểu. Nếu bỏ `_` trước thì `__10000__` thành số 10000 và trông như thật |
| 2 | 8 cột số bị lưu thành chữ vì dính `_` (`'24_'`) | Bỏ `_` ở đầu và cuối thì đọc được 100% ô còn lại | Bỏ `_` rồi ép về `float64` |
| 3 | Cột đếm có giá trị ngoài miền (tuổi −500 và 8.698, 1.499 thẻ) | Mỗi giá trị trong miền thật xuất hiện ít nhất 50 lần, ngoài miền chỉ vài lần. Kết quả không nhạy với ngưỡng từ 10 đến 100 | Giá trị ngoài miền thành NaN |
| 4 | EMI bị nhiễu | EMI thật có phần lẻ dài và không quá 1.779. Có 3.164 giá trị là số nguyên dương, lớn nhất 82.331 | EMI lớn hơn 1800 hoặc là số nguyên dương thì thành NaN |
| 5 | `Annual_Income` bị nhiễu | Thu nhập thật không đổi trong 8 tháng, giá trị nhiễu lệch hàng bậc độ lớn so với chính khách | Lệch quá 50% so với trung vị của khách thì thành NaN |
| 6 | Không dùng IQR chung cho mọi cột | IQR sẽ xóa nhầm hàng nghìn giá trị thật ở 7 cột | Chỉ áp luật cho cột có bằng chứng bị nhiễu |
| 7 | Cột không đổi theo khách (`SSN`, `Occupation`, `Name`, `Credit_Mix`, `Annual_Income`, `Interest_Rate`, `Outstanding_Debt`) | Mỗi khách chỉ có 1 giá trị, ở cả 12.500 khách | Điền bằng mode hoặc trung vị của chính khách |
| 8 | Cột có bước nhảy thật (số tài khoản, số thẻ, lương, EMI, tuổi, số lần truy vấn tín dụng) | Thay đổi diễn ra một lần (đổi việc, thêm khoản vay) | Điền bằng giá trị ở tháng gần nhất. Không dùng mode vì sẽ kéo giai đoạn này sang giai đoạn kia |
| 9 | `Num_of_Loan` bị nhiễu | Khớp 100% với số loại vay trong `Type_of_Loan` (trên 95.652 dòng hợp lệ) | Tính lại từ `Type_of_Loan` |
| 10 | `Credit_History_Age` là chuỗi chữ và có ô thiếu | Tăng đúng 1 tháng mỗi tháng ở 100% khách | Đổi ra số tháng. Ô thiếu tính bằng phép cộng (khôi phục đúng 100% ô bị che thử) |
| 11 | `Type_of_Loan` thiếu 11,4% | Thiếu đúng lúc `Num_of_Loan = 0` | Điền `No Loan`, vì đây không phải dữ liệu mất |
| 12 | `Amount_invested_monthly` và `Monthly_Balance` biến động mạnh | Phụ thuộc vào lương và thói quen chi tiêu của cùng dòng. Điền theo khách sai nhiều (NMAE 0,44 đến 0,70, đường cơ sở là 0,62 và 0,68) | Dùng mô hình học máy dự đoán tỷ lệ so với lương (tốt hơn dự đoán giá trị tuyệt đối) |
| 13 | `Payment_Behaviour` đổi ngẫu nhiên theo tháng | Không cách điền nào đoán đúng hơn ngẫu nhiên (khoảng 0,23 đến 0,28) | Giữ thành nhóm `Unknown`, không tự đoán |
| 14 | Nhãn lệch lớp, dữ liệu theo khách | Notebook 1, mục 1.2 | Chia train và test theo khách, dùng macro-F1 làm chỉ số chính |

Cột `Credit_Score` không được dùng ở bước làm sạch nào, để thông tin của nhãn không lọt vào đặc trưng.

## 4. Làm sạch: kết quả và kiểm tra

Cách điền ô thiếu được chọn bằng thí nghiệm (notebook 2, mục 4.2). Ta lấy các ô đã biết giá trị hợp lệ, che ngẫu nhiên 10%, điền lại bằng từng cách, rồi đo sai số NMAE = MAE/σ (0 là hoàn hảo, khoảng 0,8 là đoán bừa bằng trung vị). Thí nghiệm lặp với 3 hạt ngẫu nhiên. Cách tốt nhất của từng cột được so với kế hoạch khai báo trong module, và notebook dừng lại nếu hai bên không khớp.

| Nhóm cột | Cách điền được chọn | NMAE, so với điền trung vị cả cột |
|---|---|---|
| `Annual_Income`, `Interest_Rate`, `Outstanding_Debt` | trung vị của khách | 0,000 so với 0,74 đến 0,83 (khôi phục chính xác) |
| `Age`, số tài khoản, số thẻ, EMI, lương, truy vấn tín dụng | tháng gần nhất | 0,001 đến 0,067 so với 0,61 đến 0,84 |
| `Num_of_Delayed_Payment`, `Changed_Credit_Limit` | trung vị của khách | 0,12 và 0,11 so với 0,83 và 0,78 |
| `Amount_invested_monthly`, `Monthly_Balance` | mô hình dự đoán tỷ lệ so với lương | 0,052 và 0,075 (dự đoán giá trị tuyệt đối: 0,077 và 0,090; điền theo khách: 0,57 và 0,44) |
| `SSN`, `Occupation`, `Name`, `Credit_Mix` | mode của khách | đúng 100% (điền mode cả cột: đúng 0 đến 46%) |

Trước và sau (`data/reports/data_quality_report.csv`): 126.926 ô thiếu, giả hoặc vô lý (4,53% số ô) còn lại 0. Cả 12 ràng buộc kiểm tra đều đạt: không còn ô thiếu, mọi cột nằm trong miền hợp lệ, mỗi khách đúng 1 `SSN`, nghề, tên và `Credit_Mix`, `Num_of_Loan` bằng số loại vay, lịch sử tín dụng tăng 1 tháng mỗi tháng. Kết quả của notebook 2 và của hàm `clean()` trong module giống nhau từng ô, và chạy lại cho ra file CSV giống hệt từng byte.

Kiểm tra việc làm sạch có làm hỏng dữ liệu không (notebook 3):
- Phân phối của ô được điền bám sát ô gốc ở 11 trong 12 cột (KS không quá 0,08, độ lệch chuẩn không bị co lại).
- Quan hệ với nhãn được giữ nguyên: thứ tự mức liên quan không đổi, phần lớn cột đổi dưới 2%. Riêng `Credit_Mix` tăng rõ (Cramér V từ 0,40 lên 0,45). 20% số dòng mang giá trị giả nay đã được điền lại, nhưng phép so sánh trước và sau dùng hai tập dòng khác nhau nên không tách được phần tăng nào do điền lại.
- Ô được điền không tạo tín hiệu nhãn giả: tương quan Spearman với nhãn của ô được điền gần bằng ô gốc (lệch nhiều nhất 0,044, trừ `Monthly_Balance`).
- Cấu trúc tương quan giữa 16 cột số gần như không đổi (chênh trung bình 0,002).

Hạn chế của `Monthly_Balance`: có 1.209 ô thiếu (1,2%) và chúng không phải mẫu ngẫu nhiên. Lương và đầu tư ở các dòng này cao hơn nhiều, và 73% thuộc nhóm `Low_spent_Small_value` (ở các dòng còn lại là 25%). Ở vùng lương cao, sai số của mô hình điền lớn hơn khoảng 2 đến 2,5 lần nhưng vẫn nhỏ hơn nhiều so với điền theo khách (0,136 so với 0,919). Tuy vậy, với các ô thiếu thật, nó dự đoán số dư ≤ 0 cho hơn một nửa, trong khi dữ liệu gốc không có số dư nào ≤ 0. Dấu hiệu này phù hợp với việc số dư bị bỏ trống vì nó ≤ 0 (thiếu không ngẫu nhiên). Đây chỉ là giả thuyết và không kiểm chứng được. Ảnh hưởng giới hạn ở khoảng 1,2% số dòng của một cột (xem phân tích lỗi ở notebook 4).

## 5. Mô hình: việc làm sạch đóng góp gì ?

Mô hình được đặt sau cùng, chỉ để đo giá trị của việc làm sạch (notebook 4).

Cách đánh giá:
- Mỗi khách có 8 dòng gần giống nhau, nên chia ngẫu nhiên theo dòng sẽ làm cùng một người nằm ở cả train và test. Bài chia theo khách: 20% khách (2.500 người, đủ 8 tháng) làm tập kiểm tra giữ kín. 80% còn lại dùng cho mọi lựa chọn, bằng CV 5 phần chia theo khách (`StratifiedGroupKFold`).
- Nhãn lệch lớp (Standard 53%) nên chỉ số chính là macro-F1. Tập kiểm tra chỉ chấm một lần ở cuối, khoảng tin cậy 95% tính bằng bootstrap theo khách.

Thí nghiệm chính là ablation theo mức làm sạch. Cùng 21 cột gốc, cùng mô hình, cùng các phần CV, chỉ khác dữ liệu:

| Giai đoạn | Mô tả | Logistic Regression | Random Forest | HistGradientBoosting |
|---|---|---|---|---|
| S0 | Chỉ bỏ `_` và ép số, giá trị giả còn nguyên (cách "tải về là chạy") | 0,589 | 0,669 | 0,662 |
| S1 | S0, nhưng giá trị giả thành NaN, điền trung vị hoặc mode cả cột | 0,585 | 0,667 | 0,660 |
| S2 | S1, thêm bỏ giá trị vô lý, điền theo cả cột | 0,612 | 0,667 | 0,657 |
| S2* | S2, để mô hình tự xử lý NaN (không điền) | | | 0,663 |
| S3 | Làm sạch đầy đủ của bài (`credit_clean.csv`) | 0,639 | 0,679 | 0,671 |

Đây là macro-F1 trung bình trên CV 5 phần. Độ lệch chuẩn từ 0,001 đến 0,007 (xem `data/reports/ablation_cleaning.csv`).

- S3 hơn S0 ở mọi phần CV với cả ba mô hình: tăng 5,1 điểm % (Logistic Regression), 1,0 (Random Forest) và 0,9 (HistGradientBoosting). Trên tập kiểm tra giữ kín, mức tăng là 5,7 điểm % (Logistic Regression) và 0,5 điểm % (HistGradientBoosting).
- Mô hình tuyến tính được lợi nhiều nhất vì bị ngoại lệ cực đoan phá (một giá trị −3·10²⁶ làm phép chuẩn hóa mất nghĩa). Mô hình cây chịu được ngoại lệ nên lợi ích nhỏ hơn. Đây là mức đóng góp thật nhưng khiêm tốn.
- Làm sạch máy móc là chưa đủ. Với mô hình cây, chỉ bỏ giá trị giả hoặc vô lý rồi điền trung vị cả cột (S1, S2) không hơn dữ liệu thô (trong phạm vi sai số). Để mô hình tự xử lý NaN (S2*) cũng không giúp (tăng 0,1 điểm). Lợi ích chỉ xuất hiện khi điền theo từng khách và theo ràng buộc giữa các cột (S3 hơn S2* 0,8 điểm %).

Đặc trưng tự tạo chỉ đóng vai trò thứ yếu (5 nhóm, notebook 3 và mục 4.3). Thêm từng nhóm làm macro-F1 đổi trong khoảng −0,2 đến +0,4 điểm %, nằm trong sai số của CV. Nhóm duy nhất thấy được tác dụng là đặc trưng theo thời gian (`LAG`) với mô hình tuyến tính (tăng 0,4 điểm). Đặc trưng nhìn lại 8 tháng (dùng cả các tháng sau) chỉ hơn bản nhân quả vài phần mười điểm %, nên không dùng làm kết quả chính.

Mô hình cuối là Random Forest, tối ưu bằng tìm kiếm ngẫu nhiên (CV 3 phần theo khách; `max_depth=12, max_features=0,3, min_samples_leaf=9, n_estimators=159`), chạy trên dữ liệu sạch cộng đặc trưng nhân quả. Kết quả trên 2.500 khách chưa từng dùng để chọn gì:

| | accuracy | macro-F1 |
|---|---|---|
| Luôn đoán Standard (đường cơ sở) | 0,529 | 0,231 |
| Logistic Regression, dữ liệu thô | 0,617 | 0,587 |
| HistGradientBoosting, dữ liệu thô | 0,700 | 0,678 |
| HistGradientBoosting, dữ liệu sạch | 0,705 | 0,683 |
| Random Forest (đã tối ưu), dữ liệu sạch và đặc trưng nhân quả | 0,714 (CI 95%: 0,702 đến 0,726) | 0,700 (CI 95%: 0,685 đến 0,711) |

ROC-AUC (OvR, macro) là 0,861. Mô hình cuối hơn HistGradientBoosting mặc định trên dữ liệu thô 2,2 điểm % macro-F1. Trong đó 0,5 điểm là nhờ làm sạch và 1,6 điểm nhờ chọn mô hình và tối ưu siêu tham số.

Mô hình dựa vào gì và sai ở đâu (mục 4.8 và 4.9):
- Các đặc trưng quan trọng nhất theo permutation importance là `Interest_Rate`, `Credit_Mix`, `Outstanding_Debt`, `Num_Credit_Card` và độ trễ thanh toán trung bình lũy kế. Đều là chỉ số nợ và thanh toán, hợp lý về mặt tài chính và khớp với EDA. Cột định danh hay tháng không nằm trong nhóm đầu.
- Mô hình sai nhiều nhất ở các tháng mà nhãn vừa đổi so với tháng trước: accuracy 0,42, so với 0,76 khi nhãn giữ nguyên. Các dòng này chiếm 32% số lỗi dù chỉ khoảng 18% số dòng có tháng trước. Nhầm trực tiếp giữa Poor và Good chỉ chiếm 8,7% số lỗi, phần còn lại là ranh giới với Standard.
- Theo tháng: accuracy khoảng 0,79 ở tháng 1 đến 3 và khoảng 0,67 ở tháng 4 đến 8. Tỷ lệ nhãn đổi nhảy lên khoảng 26 đến 27% ở tháng 4 và tháng 7 (các tháng khác 13 đến 15%), giải thích được một phần. Phần còn lại chưa xác định, có thể do cách dữ liệu tổng hợp được tạo ra. Theo `Credit_Mix`: 0,67 (Bad), 0,83 (Standard), 0,57 (Good), nhóm Good khó nhất.
- Dòng phải sửa càng nhiều ô thì accuracy giảm nhẹ (0,720 xuống 0,699 từ 0 đến từ 3 ô), nhưng mức giảm chưa đạt ý nghĩa thống kê. Với 228 dòng có `Monthly_Balance` phải điền thì không kết luận được gì.

## 6. Dự đoán test.csv và file output

`test.csv` không có nhãn, và kiểm tra cấu trúc cho thấy đây không phải khách mới. File có 50.000 dòng của đúng 12.500 khách trong `train.csv`, mỗi khách 4 tháng (September đến December), nối tiếp tháng 8 của train. ID dòng không trùng với train. Mức nhiễu giống train (ví dụ `Credit_Mix` bị thay bằng `_` ở 19,6% dòng test và 20,2% dòng train), và `Credit_History_Age` của mỗi khách tăng đúng 1 tháng mỗi tháng xuyên suốt 12 tháng ở cả 12.500 khách. Vì vậy các quy tắc làm sạch của train dùng lại được cho test.

`python3 make_submission.py` làm các bước sau:

1. Nối train (bỏ cột nhãn ngay lúc nối) và test thành chuỗi 12 tháng của mỗi khách, rồi làm sạch bằng đúng các bước của notebook 2 (`clean_panel` trong `src/credit_submission.py`). Khi đưa vào dữ liệu 8 tháng của train, hàm này cho kết quả giống hệt `cc.clean()` (đã kiểm tra, 0 ô khác). Kết quả đạt cả 12 ràng buộc kiểm tra.
2. Tạo đặc trưng cho các dòng test. Đặc trưng theo thời gian chỉ nhìn tháng hiện tại và các tháng trước, nên tháng 9 nhìn được lịch sử từ tháng 1 đến 8. Ma trận đặc trưng không chứa nhãn hay cột định danh (`ID`, `Customer_ID`, `Name`, `SSN`), có `assert` kiểm tra. `Name` và `SSN` vẫn được làm sạch trong `credit_clean.csv` vì bài chấm phần làm sạch, nhưng không bao giờ là đặc trưng.
3. Lấy mô hình và siêu tham số đã chọn ở notebook 4 (đọc từ `data/reports/final_test_metrics.json`, chọn bằng CV chia theo khách), huấn luyện lại trên toàn bộ 100.000 dòng train rồi dự đoán 50.000 dòng test. Nhãn mã hóa Poor = 0, Standard = 1, Good = 2 khi huấn luyện và đổi lại thành chữ khi xuất (dự đoán giống hệt khi huấn luyện bằng nhãn chữ).
4. Ghi `outputs/submission.csv`, đọc lại từ đĩa và kiểm tra 7 điều: đúng hai cột, không có cột index, đúng số dòng, `ID` giống `test.csv` và cùng thứ tự, không mất hay trùng `ID`, nhãn chỉ gồm Poor, Standard, Good, không có ô trống. Chạy hai lần cho file giống hệt từng byte.

Nhãn dự đoán trên test: Standard 50,5%, Poor 26,7%, Good 22,8% (train: 53,2%, 29,0%, 17,8%). Cũng mô hình này, khi chấm trên khách giữ kín ở notebook 4, dự đoán Good khoảng 22 đến 24% mỗi tháng, trong khi tỷ lệ Good thật của các khách đó tăng theo tháng (khoảng 16% ở tháng 1 lên 21% ở tháng 8). Tỷ lệ dự đoán trên test khớp với hành vi này.

**Điểm của file nộp bài.** Test không có nhãn nên không tính được accuracy cho file này ở máy; điểm thật chỉ có khi nộp lên Kaggle. Hai con số sau chỉ cho biết cỡ độ lớn:
- Khách giữ kín của notebook 4 (khách mới, đủ 8 tháng): accuracy 0,714, macro-F1 0,700.
- Mô phỏng đúng tình huống của test (`check_robustness.py`): huấn luyện trên tháng 1 đến 4 của mọi khách, chấm trên tháng 5 đến 8 của cùng khách. Accuracy 0,684, macro-F1 0,669, đường cơ sở (luôn đoán Standard) 0,517. Mô hình ở đây chỉ có 4 tháng để học (bản thật có 8 tháng), và các tháng sau khó hơn (notebook 4, mục 4.9: accuracy khoảng 0,79 ở tháng 1 đến 3 và 0,67 ở tháng 4 đến 8). Cách chia này với khách chưa từng thấy cho 0,663 và 0,648, thấp hơn 2 điểm % so với cùng khách.

Các kiểm tra bổ sung của `check_robustness.py` (kết quả đầy đủ ở `data/reports/robustness_checks.json`):

| Kiểm tra | Kết quả |
|---|---|
| Chia CV: có khách nào nằm cả ở train và validation không | Không, ở cả CV 5 phần và CV 3 phần (dùng để chỉnh tham số). Tỷ lệ nhãn của mỗi phần lệch tối đa 0,02 điểm % so với toàn bộ |
| Học lại mô hình điền thiếu của `Amount_invested_monthly` và `Monthly_Balance` chỉ trên phần train của từng phần CV | macro-F1 chênh trung bình −0,01 điểm % (từng phần từ −0,13 đến +0,19), nằm trong sai số |
| Thêm 3 đặc trưng `Monthly_Gross_Income`, `Debt_to_Income_Ratio`, `Monthly_Savings` | macro-F1 0,6898 so với 0,6907 của bộ chính, tốt hơn ở 2 trong 5 phần. Thêm riêng 2 đặc trưng mới thì 0,6902. Quy tắc đặt trước là chỉ dùng nếu cao hơn ở cả 5 phần, nên không dùng. `Debt_to_Income_Ratio` trùng `Debt_to_Income` đã có (tương quan 1,0) |
| Chỉ số theo lớp của mô hình cuối trong CV 5 phần (precision, recall, F1) | Poor 0,731, 0,659, 0,693. Standard 0,757, 0,741, 0,749. Good 0,572, 0,702, 0,630. macro-F1 0,6907 ± 0,0032, accuracy 0,7102 ± 0,0029 |

## 7. Hạn chế

1. Dữ liệu là dữ liệu tổng hợp. Nhãn có nhiều nhiễu theo tháng, nên mọi mô hình dừng quanh 0,70 đến 0,71 accuracy trong CV, kể cả khi biết trước cả 8 tháng. Điều đáng tin ở bài này là chênh lệch giữa các mức làm sạch trong cùng điều kiện, không phải con số tuyệt đối. Không nên suy ra hiệu năng trên dữ liệu ngân hàng thật.
2. Thí nghiệm che 10% giả định các ô thiếu thật xuất hiện ngẫu nhiên. Với `Monthly_Balance` điều này không đúng: các ô thiếu thuộc nhóm khách khác và có dấu hiệu thiếu không ngẫu nhiên (chỉ là giả thuyết). Giá trị điền cho `Amount_invested_monthly` và `Monthly_Balance` là ước lượng, không phải số thật. 
3. Miền hợp lệ được suy ra từ tần suất, nên có thể loại nhầm vài giá trị thật hiếm sát biên. Kết quả không nhạy với ngưỡng từ 10 đến 100 nhưng cũng không tuyệt đối.
4. `Name` còn nhiễu chữ cái (nhiều tên có một chữ thường dính ở cuối, ví dụ `Ashleyh`). Không sửa vì cột này không dùng làm đặc trưng.
5. Các tham số làm sạch được tính trên cả 100.000 dòng (không dùng nhãn) trước khi chia train và test. Phần duy nhất học xuyên khách là mô hình điền thiếu của hai cột tiền, còn lại đều tính trong từng khách. Học lại mô hình đó chỉ trên phần train của từng phần CV không đổi kết quả (macro-F1 chênh −0,01 điểm %, mục 6). Khi dự đoán test, làm sạch chạy trên chuỗi train cộng test (vẫn không dùng nhãn), nên mô hình điền thiếu có nhìn đặc trưng của test.
6. Làm sạch dùng cả các tháng sau của cùng khách để sửa một ô (điền theo tháng gần nhất, trung vị của khách, suy ra lịch sử tín dụng). Đây là sửa lại bản ghi đã có đủ cả kỳ, không phải dự báo. Nếu triển khai theo thời gian thực thì chưa có các tháng sau. Riêng đặc trưng đưa vào mô hình chính thì chỉ nhìn tháng hiện tại và các tháng trước.
7. Chỉ chia train và test một lần. Tìm kiếm siêu tham số có giới hạn (20 và 8 cấu hình, CV 3 phần).
8. Điểm trên `test.csv` không đo được ở máy vì file không có nhãn. CV ở notebook 4 chia theo khách (khách mới), còn `test.csv` là cùng khách ở các tháng sau, nên hai con số không so trực tiếp được. Mô phỏng chia theo thời gian cho accuracy 0,684 (mục 6), thấp hơn 0,714 của khách giữ kín.

## 8. Chạy lại

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn nbformat nbclient ipykernel
# đặt train.csv và test.csv (tải từ Kaggle) vào thư mục data/
bash run_all.sh
```

`run_all.sh` chạy lần lượt notebook 1 đến 4, mất khoảng 1 đến 1,5 giờ trên 2 CPU, phần lớn là notebook 4, rồi chạy `make_submission.py` (khoảng 2 phút) để tạo `outputs/submission.csv`. Cũng có thể mở từng notebook và chạy từ đầu đến cuối (Restart and Run All).

Chỉ cần tạo lại file output (đã có `data/credit_clean.csv`) thì chạy `python3 make_submission.py`. Các kiểm tra bổ sung chạy riêng bằng `python3 check_robustness.py` (khoảng 20 phút, có thể chọn phần chạy bằng `--checks`).

Kết quả tái lập được: chạy lại notebook 2 cho ra `credit_clean.csv` giống hệt từng byte, chạy lại `make_submission.py` cho `submission.csv` giống hệt từng byte, và các hạt ngẫu nhiên đều được cố định (ghi trong `data/reports/submission_log.json`).

Code làm sạch cũng dùng riêng được như một module:

```python
import sys
sys.path.insert(0, 'src')
import pandas as pd
import credit_cleaning as cc

clean_df = cc.clean(pd.read_csv('data/train.csv', low_memory=False))
assert all(cc.validate_clean(clean_df).values())
```
