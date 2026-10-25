# Phân tích, làm sạch dữ liệu và phân loại điểm tín dụng khách hàng

Môn Khai phá dữ liệu, Đại học Phenikaa. Sinh viên: Lê Nguyễn Hải Anh.

Dữ liệu: [Credit Score Classification (Kaggle)](https://www.kaggle.com/datasets/parisrohan/credit-score-classification), gồm `train.csv` (có nhãn) và `test.csv` (không có nhãn).

## 1. Bài này làm gì

Bài toán là đoán `Credit_Score` của một khách hàng ngân hàng ở một tháng, có ba mức Poor, Standard và Good, dựa vào hồ sơ tài chính của khách trong tháng đó.

File train có 100.000 dòng, tức 12.500 khách, mỗi khách 8 tháng. Dữ liệu này rất bẩn:

- Có giá trị giả như `_______`, `#F%$D@*&8`, `__10000__`.
- Nhiều số bị dính dấu `_` ở đầu hoặc cuối.
- Có giá trị vô lý: tuổi −500 và 8.698, 1.499 thẻ tín dụng, lãi suất 5.797%, EMI 82.331.
- Có ô thiếu, và các tháng của cùng một khách đôi khi mâu thuẫn với nhau.

Vì vậy phần chính của bài là phân tích và làm sạch dữ liệu. Mô hình học máy để ở cuối, chỉ để đo xem làm sạch giúp được bao nhiêu.

Mỗi bước là một notebook, và mỗi quyết định đều dựa trên số đo được:

```
train.csv (dữ liệu bẩn)
   notebook 1: xem dữ liệu, phân tích khám phá (EDA)
   notebook 2: làm sạch                              ->  credit_clean.csv
   notebook 3: kiểm tra dữ liệu sạch, tạo đặc trưng  ->  credit_features.csv
   notebook 4: mô hình và đánh giá

test.csv (không có nhãn)
   make_output.py: làm sạch, tạo đặc trưng, dự đoán  ->  outputs/output.csv
```

## 2. Có gì trong thư mục

| Đường dẫn | Nội dung |
|---|---|
| `notebooks/1_data_understanding_eda.ipynb` | Xem cấu trúc dữ liệu (khách và tháng), đo từng loại lỗi, rồi lập bảng lỗi, bằng chứng và quyết định xử lý |
| `notebooks/2_data_cleaning.ipynb` | Làm sạch theo 6 bước. Cách điền ô thiếu được chọn bằng thí nghiệm che giá trị. Có 12 điều kiện kiểm tra sau khi làm sạch, báo cáo chất lượng trước và sau, và xuất `credit_clean.csv` |
| `notebooks/3_post_cleaning_features.ipynb` | Kiểm tra làm sạch có làm méo dữ liệu không, tạo 5 nhóm đặc trưng và đo xem chúng có ích không. Xuất `credit_features.csv` |
| `notebooks/4_modeling_evaluation.ipynb` | Thí nghiệm so sánh các mức làm sạch, so sánh mô hình, chỉnh siêu tham số, kết quả cuối trên tập giữ kín, đặc trưng quan trọng, phân tích lỗi |
| `src/credit_cleaning.py` | Code làm sạch, có hàm `clean()` chạy cả quy trình. Notebook 2 import và in code từ file này, nên code hiện trong notebook chính là code đang chạy |
| `src/credit_features.py` | Code tạo đặc trưng |
| `src/notebook_utils.py` | Hàm dùng chung cho các notebook (in code, màu biểu đồ, vài hàm thống kê) |
| `src/credit_output.py` | Hàm cho bước dự đoán test: làm sạch chuỗi 12 tháng, dựng ma trận đặc trưng, kiểm tra file output |
| `make_output.py` | Tạo `outputs/output.csv` cho `test.csv` (mục 6) |
| `check_robustness.py` | Các kiểm tra thêm: cách chia CV, học lại mô hình điền thiếu theo từng phần CV, thử thêm ba đặc trưng, chia theo thời gian (mục 6) |
| `data/train.csv` | Dữ liệu gốc tải từ Kaggle, không đưa lên kho mã |
| `data/test.csv` | File test tải từ Kaggle, không đưa lên kho mã. Có 50.000 dòng, là 4 tháng sau (September đến December) của đúng 12.500 khách trong train |
| `data/credit_clean.csv` | Dữ liệu sau làm sạch, 100.000 dòng, 29 cột, không còn ô thiếu, giá trị giả hay giá trị vô lý |
| `data/credit_features.csv` | Dữ liệu sạch cộng 45 đặc trưng tự tạo, chạy lại notebook 3 là sinh ra |
| `data/credit_test_clean.csv` | 50.000 dòng của `test.csv` sau làm sạch |
| `outputs/output.csv` | File output, hai cột `ID` và `Credit_Score`, đủ 50.000 dòng theo thứ tự của `test.csv` |
| `data/reports/` | Báo cáo chất lượng dữ liệu trước và sau, nhật ký làm sạch, các bảng kết quả mô hình, `output_log.json` (cấu hình, seed, kết quả kiểm tra file output) và `robustness_checks.json` |
| `run_all.sh`, `run.py` | Chạy 4 notebook từ đầu đến cuối, rồi tạo file output nếu có `test.csv` |

Cách chạy lại ở mục 8.

## 3. Phân tích dữ liệu và các quyết định làm sạch

Dữ liệu này là 12.500 khách chứ không phải 100.000 quan sát độc lập. Nhãn đổi chậm theo tháng và bị lệch (Standard 53%, Poor 29%, Good 18%). Từ đó có hai điều cần nhớ:

1. Các tháng khác của chính khách đó là nguồn tốt nhất để sửa một ô lỗi.
2. Phải chia train và test theo khách, không thì cùng một người sẽ nằm ở cả hai tập.

Dữ liệu bẩn cỡ nào (notebook 1, mục 1.9): 135.774 trên 2.800.000 ô (4,8%) có vấn đề. Có 76,3% số dòng có ít nhất một ô lỗi, và khách nào cũng có lỗi. Xóa dòng lỗi không làm được. Sau bước 1 đến 3, nếu xóa các dòng còn ô trống thì chỉ còn 29,5% dữ liệu và không còn khách nào đủ 8 tháng. Nên bài chọn sửa hoặc khôi phục từng ô.

Bảng dưới tóm tắt từng vấn đề, bằng chứng và cách xử lý (chi tiết và biểu đồ ở notebook 1, mục 1.10):

| # | Vấn đề | Bằng chứng | Cách xử lý |
|---|---|---|---|
| 1 | 7 cột có giá trị giả (`#F%$D@*&8`, `_______`, `_`, `!@9#%8`, `__10000__`, `__-333…333__`) | Tìm bằng 3 cách độc lập: không có chữ cái, sai định dạng, lặp lại bất thường hoặc quá lớn so với thực tế. Nhóm giá trị giả không mang thông tin về nhãn | Đổi thành NaN trước khi ép kiểu. Nếu bỏ `_` trước thì `__10000__` thành số 10000 và trông như số thật |
| 2 | 8 cột số bị lưu thành chữ vì dính `_` (ví dụ `'24_'`) | Bỏ `_` hai đầu thì đọc được 100% ô còn lại | Bỏ `_` rồi ép về `float64` |
| 3 | Cột đếm có giá trị ngoài miền (tuổi −500 và 8.698, 1.499 thẻ) | Mỗi giá trị trong miền thật xuất hiện ít nhất 50 lần, giá trị ngoài miền chỉ vài lần. Đổi ngưỡng từ 10 đến 100 thì kết quả gần như không đổi | Ngoài miền thì thành NaN |
| 4 | EMI bị rác | EMI thật là số thực có phần lẻ dài và không quá 1.779. Có 3.164 giá trị là số nguyên dương, lớn nhất 82.331 | EMI lớn hơn 1800 hoặc là số nguyên dương thì thành NaN |
| 5 | `Annual_Income` bị rác | Thu nhập thật không đổi trong 8 tháng, còn giá trị rác lệch rất xa so với chính khách đó | Lệch quá 50% so với trung vị của khách thì thành NaN |
| 6 | Không dùng một luật IQR chung cho mọi cột | IQR sẽ xóa nhầm hàng nghìn giá trị thật ở 7 cột | Chỉ loại giá trị ở những cột có bằng chứng bị rác |
| 7 | Cột không đổi theo khách (`SSN`, `Occupation`, `Name`, `Credit_Mix`, `Annual_Income`, `Interest_Rate`, `Outstanding_Debt`) | Mỗi khách chỉ có 1 giá trị, đúng với cả 12.500 khách | Điền bằng mode hoặc trung vị của chính khách |
| 8 | Cột có bước nhảy thật (số tài khoản, số thẻ, lương, EMI, tuổi, số lần truy vấn tín dụng) | Đây là thay đổi thật, không phải lỗi: chỉ xảy ra một lần và đi theo cặp (289 khách đổi cả số tài khoản lẫn số thẻ, 741 khách đổi cả lương lẫn EMI) | Điền bằng giá trị ở tháng gần nhất. Không dùng mode vì sẽ kéo giai đoạn này sang giai đoạn kia |
| 9 | `Num_of_Loan` bị rác | Bằng đúng số loại vay trong `Type_of_Loan` ở 100% trong 95.652 dòng hợp lệ | Tính lại từ `Type_of_Loan` |
| 10 | `Credit_History_Age` là chuỗi chữ và có ô thiếu | Tăng đúng 1 tháng mỗi tháng ở 100% khách | Đổi ra số tháng. Ô thiếu tính bằng phép cộng (che thử 10% thì tính lại đúng 100%) |
| 11 | `Type_of_Loan` thiếu 11,4% | Thiếu đúng ở những khách có `Num_of_Loan = 0` | Điền `No Loan`, vì đây không phải dữ liệu bị mất |
| 12 | `Amount_invested_monthly` và `Monthly_Balance` biến động mạnh | Phụ thuộc vào lương và chi tiêu của cùng dòng. Điền theo khách sai khá nhiều (NMAE từ 0,44 đến 0,70, trong khi điền trung vị cả cột là 0,62 và 0,68) | Dùng mô hình học máy dự đoán tỷ lệ so với lương, cách này tốt hơn dự đoán thẳng giá trị |
| 13 | `Payment_Behaviour` đổi theo tháng | Không cách điền nào đoán tốt hơn đoán bừa (độ chính xác 0,23 đến 0,28) | Giữ thành nhóm riêng `Unknown`, không tự đoán |
| 14 | Nhãn lệch lớp, dữ liệu theo khách | Notebook 1, mục 1.2 | Chia train và test theo khách, dùng macro-F1 làm chỉ số chính |

Cột `Credit_Score` không được dùng ở bước làm sạch nào, để thông tin của nhãn không lọt vào đặc trưng.

## 4. Làm sạch: kết quả và cách kiểm tra

Cách điền ô thiếu của từng cột được chọn bằng thí nghiệm (notebook 2, mục 4.2). Lấy các ô đã biết giá trị đúng, che ngẫu nhiên 10%, điền lại bằng từng cách rồi đo sai số NMAE = MAE/σ. NMAE bằng 0 là hoàn hảo, khoảng 0,8 là ngang với việc điền bằng trung vị. Thí nghiệm lặp với 3 hạt ngẫu nhiên. Cách tốt nhất của mỗi cột được đối chiếu với kế hoạch khai báo trong module, nếu hai bên lệch nhau thì notebook dừng lại.

| Nhóm cột | Cách điền được chọn | NMAE, so với điền trung vị cả cột |
|---|---|---|
| `Annual_Income`, `Interest_Rate`, `Outstanding_Debt` | trung vị của khách | 0,000 so với 0,74 đến 0,83 (khôi phục chính xác) |
| `Age`, số tài khoản, số thẻ, EMI, lương, số lần truy vấn tín dụng | tháng gần nhất | 0,001 đến 0,067 so với 0,61 đến 0,84 |
| `Num_of_Delayed_Payment`, `Changed_Credit_Limit` | trung vị của khách | 0,12 và 0,11 so với 0,83 và 0,78 |
| `Amount_invested_monthly`, `Monthly_Balance` | mô hình dự đoán tỷ lệ so với lương | 0,052 và 0,075 (dự đoán thẳng giá trị: 0,077 và 0,090; điền theo khách: 0,57 và 0,44) |
| `SSN`, `Occupation`, `Name`, `Credit_Mix` | mode của khách | đúng 100% (điền mode cả cột chỉ đúng 0 đến 46%) |

Trước và sau khi làm sạch (`data/reports/data_quality_report.csv`): 126.926 ô thiếu, giả hoặc vô lý (4,53% số ô) còn lại 0. Cả 12 điều kiện kiểm tra đều đạt, ví dụ không còn ô thiếu, mọi cột nằm trong miền hợp lệ, mỗi khách đúng 1 `SSN`, nghề, tên và `Credit_Mix`, `Num_of_Loan` bằng số loại vay, lịch sử tín dụng tăng 1 tháng mỗi tháng. Kết quả chạy từng bước ở notebook 2 và kết quả của hàm `clean()` giống nhau từng ô. Chạy lại cho ra file CSV giống hệt từng byte.

Notebook 3 kiểm tra xem làm sạch có làm hỏng dữ liệu không:

- Phân phối của ô được điền bám sát ô gốc ở 11 trong 12 cột (KS không quá 0,08, độ lệch chuẩn không bị co lại).
- Quan hệ với nhãn gần như giữ nguyên: thứ tự mức liên quan không đổi, phần lớn cột đổi dưới 2%. `Credit_Mix` tăng rõ (Cramér V từ 0,40 lên 0,45). Lý do có thể là 20% số dòng từng mang giá trị giả nay đã được điền lại. Tuy vậy phép so sánh trước và sau dùng hai tập dòng khác nhau, nên chưa tách được phần tăng nào là do điền lại.
- Ô được điền không tạo ra tín hiệu nhãn giả: tương quan Spearman với nhãn của ô được điền gần bằng ô gốc (lệch nhiều nhất 0,044, trừ `Monthly_Balance`).
- Tương quan giữa 16 cột số gần như không đổi (chênh trung bình 0,002).

Điểm yếu của `Monthly_Balance`: cột này có 1.209 ô thiếu (1,2%) và chúng không thiếu ngẫu nhiên. Lương và số tiền đầu tư ở các dòng này cao hơn nhiều, và 73% thuộc nhóm `Low_spent_Small_value` (các dòng còn lại là 25%). Ở vùng lương cao, sai số của mô hình điền lớn hơn khoảng 2 đến 2,5 lần, nhưng vẫn nhỏ hơn nhiều so với điền theo khách (0,136 so với 0,919). Với các ô thiếu thật, mô hình dự đoán số dư ≤ 0 cho hơn một nửa, trong khi dữ liệu gốc không có số dư nào ≤ 0. Có thể số dư bị bỏ trống chính vì nó ≤ 0, nhưng đây chỉ là giả thuyết và không kiểm chứng được. Ảnh hưởng chỉ nằm ở khoảng 1,2% số dòng của một cột (xem phân tích lỗi ở notebook 4).

## 5. Mô hình: làm sạch giúp được bao nhiêu

Mô hình đặt sau cùng, chỉ để đo giá trị của việc làm sạch (notebook 4).

Cách đánh giá:

- Mỗi khách có 8 dòng gần giống nhau. Nếu chia ngẫu nhiên theo dòng thì cùng một người sẽ nằm ở cả train và test. Bài chia theo khách: 20% khách (2.500 người, đủ 8 tháng) làm tập kiểm tra giữ kín. 80% còn lại dùng cho mọi lựa chọn, bằng CV 5 phần chia theo khách (`StratifiedGroupKFold`).
- Nhãn lệch (Standard 53%) nên chỉ số chính là macro-F1. Tập kiểm tra chỉ chấm một lần ở cuối. Khoảng tin cậy 95% tính bằng bootstrap theo khách.

Thí nghiệm chính là so các mức làm sạch với nhau. Cùng 21 cột gốc, cùng mô hình, cùng các phần CV, chỉ khác dữ liệu:

| Giai đoạn | Mô tả | Logistic Regression | Random Forest | HistGradientBoosting |
|---|---|---|---|---|
| S0 | Chỉ bỏ `_` và ép số, giá trị giả còn nguyên (tải về là chạy) | 0,589 | 0,669 | 0,662 |
| S1 | Như S0, nhưng giá trị giả thành NaN rồi điền trung vị hoặc mode cả cột | 0,585 | 0,667 | 0,660 |
| S2 | Như S1, thêm bỏ giá trị vô lý, vẫn điền theo cả cột | 0,612 | 0,667 | 0,657 |
| S2* | Như S2, nhưng để mô hình tự xử lý NaN (không điền) | | | 0,663 |
| S3 | Làm sạch đầy đủ của bài (`credit_clean.csv`) | 0,639 | 0,679 | 0,671 |

Đây là macro-F1 trung bình trên CV 5 phần. Độ lệch chuẩn từ 0,001 đến 0,007 (xem `data/reports/ablation_cleaning.csv`).

- S3 cao hơn S0 ở mọi phần CV với cả ba mô hình. Mức tăng là 5,1 điểm % (Logistic Regression), 1,0 (Random Forest) và 0,9 (HistGradientBoosting). Trên tập kiểm tra giữ kín, mức tăng là 5,7 điểm % (Logistic Regression) và 0,5 điểm % (HistGradientBoosting).
- Logistic Regression được lợi nhiều nhất vì nó bị các giá trị cực lớn làm hỏng (một giá trị −3·10²⁶ làm bước chuẩn hóa mất tác dụng). Mô hình cây ít bị ảnh hưởng hơn nên mức tăng nhỏ hơn. Vậy làm sạch có giúp thật, nhưng không nhiều.
- Làm sạch kiểu cho có thì chưa đủ. Với mô hình cây, chỉ bỏ giá trị giả hoặc vô lý rồi điền trung vị cả cột (S1, S2) không hơn dữ liệu thô, trong phạm vi sai số. Để mô hình tự xử lý NaN (S2*) cũng không giúp (tăng 0,1 điểm). Chỉ khi điền theo từng khách và theo quan hệ giữa các cột (S3) mới thấy lợi, S3 hơn S2* 0,8 điểm %.

Đặc trưng tự tạo chỉ đóng góp thêm một chút (5 nhóm, notebook 3 và notebook 4 mục 4.3). Thêm từng nhóm làm macro-F1 đổi trong khoảng −0,2 đến +0,4 điểm %, nằm trong sai số của CV. Chỉ có đặc trưng theo thời gian (`LAG`) thấy được tác dụng, ở mô hình tuyến tính (tăng 0,4 điểm). Đặc trưng nhìn lại cả 8 tháng (dùng cả các tháng sau) chỉ hơn bản chỉ nhìn tháng trước vài phần mười điểm %, nên không dùng làm kết quả chính.

Mô hình cuối là Random Forest, chỉnh siêu tham số bằng tìm kiếm ngẫu nhiên (CV 3 phần chia theo khách; `max_depth=12, max_features=0,3, min_samples_leaf=9, n_estimators=159`), chạy trên dữ liệu sạch cộng đặc trưng chỉ nhìn tháng hiện tại và các tháng trước. Kết quả trên 2.500 khách chưa dùng để chọn gì:

| | accuracy | macro-F1 |
|---|---|---|
| Luôn đoán Standard (để so) | 0,529 | 0,231 |
| Logistic Regression, dữ liệu thô | 0,617 | 0,587 |
| HistGradientBoosting, dữ liệu thô | 0,700 | 0,678 |
| HistGradientBoosting, dữ liệu sạch | 0,705 | 0,683 |
| Random Forest (đã chỉnh), dữ liệu sạch và đặc trưng nhìn tháng trước | 0,714 (CI 95%: 0,702 đến 0,726) | 0,700 (CI 95%: 0,685 đến 0,711) |

ROC-AUC (OvR, macro) là 0,861. Mô hình cuối hơn HistGradientBoosting mặc định trên dữ liệu thô 2,2 điểm % macro-F1, trong đó 0,5 điểm nhờ làm sạch và 1,6 điểm nhờ chọn mô hình và chỉnh siêu tham số.

Mô hình dựa vào gì và sai ở đâu (notebook 4, mục 4.8 và 4.9):

- Đặc trưng quan trọng nhất theo permutation importance là `Interest_Rate`, `Credit_Mix`, `Outstanding_Debt`, `Num_Credit_Card` và độ trễ thanh toán trung bình tính dồn. Toàn là chỉ số về nợ và thanh toán, hợp lý với kiến thức tài chính và khớp với phần EDA. Cột định danh hay tháng không nằm trong nhóm đầu.
- Mô hình sai nhiều nhất ở các tháng mà nhãn vừa đổi so với tháng trước: accuracy 0,42, so với 0,76 khi nhãn giữ nguyên. Các dòng này chiếm 32% số lỗi dù chỉ khoảng 18% số dòng có tháng trước. Nhầm thẳng giữa Poor và Good chỉ chiếm 8,7% số lỗi, phần lớn lỗi nằm ở ranh giới với Standard.
- Theo tháng: accuracy khoảng 0,79 ở tháng 1 đến 3 và khoảng 0,67 ở tháng 4 đến 8. Tỷ lệ nhãn đổi nhảy lên khoảng 26 đến 27% ở tháng 4 và tháng 7 (các tháng khác là 13 đến 15%), giải thích được một phần. Phần còn lại chưa rõ nguyên nhân, có thể do cách dữ liệu tổng hợp được tạo ra. Theo `Credit_Mix`: 0,67 (Bad), 0,83 (Standard), 0,57 (Good), nhóm Good khó đoán nhất.
- Dòng phải sửa càng nhiều ô thì accuracy giảm nhẹ (từ 0,720 xuống 0,699 khi phải sửa từ 3 ô), nhưng mức giảm chưa đủ để kết luận về mặt thống kê. Với 228 dòng có `Monthly_Balance` phải điền thì không kết luận được gì.

## 6. Dự đoán test.csv và file output

`test.csv` không có nhãn, và kiểm tra cấu trúc cho thấy đây không phải khách mới. File có 50.000 dòng của đúng 12.500 khách trong `train.csv`, mỗi khách 4 tháng (September đến December), nối tiếp tháng 8 của train. ID dòng không trùng với train. Mức độ bẩn giống train (ví dụ `Credit_Mix` bị thay bằng `_` ở 19,6% dòng test và 20,2% dòng train), và `Credit_History_Age` của mỗi khách tăng đúng 1 tháng mỗi tháng suốt 12 tháng, đúng với cả 12.500 khách. Nên các quy tắc làm sạch của train dùng lại được cho test.

`python3 make_output.py` (khoảng 2 phút) làm các bước sau:

1. Nối train (bỏ cột nhãn ngay lúc nối) với test thành chuỗi 12 tháng của mỗi khách, rồi làm sạch bằng đúng các bước của notebook 2 (hàm `clean_panel` trong `src/credit_output.py`). Khi đưa vào dữ liệu 8 tháng của train, hàm này cho kết quả giống hệt `cc.clean()` (đã kiểm tra, 0 ô khác). Kết quả đạt cả 12 điều kiện kiểm tra.
2. Tạo đặc trưng cho các dòng test. Đặc trưng theo thời gian chỉ nhìn tháng hiện tại và các tháng trước, nên tháng 9 nhìn được lịch sử từ tháng 1 đến tháng 8. Ma trận đặc trưng không chứa nhãn hay cột định danh (`ID`, `Customer_ID`, `Name`, `SSN`), có `assert` kiểm tra. `Name` và `SSN` vẫn được làm sạch trong `credit_clean.csv` vì bài chấm phần làm sạch, nhưng không bao giờ làm đặc trưng.
3. Lấy mô hình và siêu tham số đã chọn ở notebook 4 (đọc từ `data/reports/final_test_metrics.json`), huấn luyện lại trên toàn bộ 100.000 dòng train rồi dự đoán 50.000 dòng test. Khi huấn luyện, nhãn được mã hóa Poor = 0, Standard = 1, Good = 2 và đổi lại thành chữ khi xuất file (kết quả giống hệt khi huấn luyện bằng nhãn chữ).
4. Ghi `outputs/output.csv`, đọc lại từ đĩa và kiểm tra 7 điều: đúng hai cột, không có cột index, đúng số dòng, `ID` giống `test.csv` và cùng thứ tự, không mất hay trùng `ID`, nhãn chỉ gồm Poor, Standard, Good, và không có ô trống. Chạy hai lần cho file giống hệt từng byte.

Nhãn dự đoán trên test: Standard 50,5%, Poor 26,7%, Good 22,8% (train: 53,2%, 29,0%, 17,8%). Cũng mô hình này, khi chấm trên khách giữ kín ở notebook 4, dự đoán Good khoảng 22 đến 24% mỗi tháng, trong khi tỷ lệ Good thật của các khách đó tăng dần theo tháng (khoảng 16% ở tháng 1 lên 21% ở tháng 8). Tỷ lệ dự đoán trên test khớp với xu hướng này.

**Điểm của file output.** Test không có nhãn nên không tính được accuracy cho file này ở máy. Điểm thật chỉ có khi nộp lên Kaggle. Hai con số dưới đây chỉ cho biết cỡ độ lớn:

- Khách giữ kín của notebook 4 (khách mới, đủ 8 tháng): accuracy 0,714, macro-F1 0,700.
- Mô phỏng đúng tình huống của test (`check_robustness.py`): huấn luyện trên tháng 1 đến 4 của mọi khách, chấm trên tháng 5 đến 8 của cùng những khách đó. Accuracy 0,684, macro-F1 0,669, còn đoán Standard cho tất cả chỉ được 0,517. Mô hình ở đây chỉ có 4 tháng để học (bản thật có 8 tháng), và các tháng sau khó đoán hơn (notebook 4, mục 4.9: accuracy khoảng 0,79 ở tháng 1 đến 3 và 0,67 ở tháng 4 đến 8). Nếu chia theo thời gian mà lại với khách chưa từng thấy thì được 0,663 và 0,648, thấp hơn 2 điểm % so với cùng khách.

Các kiểm tra thêm của `check_robustness.py` (đầy đủ ở `data/reports/robustness_checks.json`):

| Kiểm tra | Kết quả |
|---|---|
| Chia CV: có khách nào nằm cả ở train và validation không | Không, ở cả CV 5 phần và CV 3 phần (dùng để chỉnh tham số). Tỷ lệ nhãn của mỗi phần lệch tối đa 0,02 điểm % so với toàn bộ |
| Học lại mô hình điền thiếu của `Amount_invested_monthly` và `Monthly_Balance` chỉ trên phần train của từng phần CV | macro-F1 chênh trung bình −0,01 điểm % (từng phần từ −0,13 đến +0,19), nằm trong sai số |
| Thêm 3 đặc trưng `Monthly_Gross_Income`, `Debt_to_Income_Ratio`, `Monthly_Savings` | macro-F1 0,6898 so với 0,6907 của bộ chính, tốt hơn ở 2 trong 5 phần. Thêm riêng 2 đặc trưng mới thì được 0,6902. Quy tắc đặt trước là chỉ dùng nếu cao hơn ở cả 5 phần, nên không dùng. `Debt_to_Income_Ratio` trùng với `Debt_to_Income` đã có (tương quan 1,0) |
| Chỉ số theo lớp của mô hình cuối trong CV 5 phần (precision, recall, F1) | Poor 0,731, 0,659, 0,693. Standard 0,757, 0,741, 0,749. Good 0,572, 0,702, 0,630. macro-F1 0,6907 ± 0,0032, accuracy 0,7102 ± 0,0029 |

## 7. Hạn chế

1. Dữ liệu là dữ liệu tổng hợp. Nhãn nhiễu nhiều theo tháng, nên mọi mô hình đều dừng quanh 0,70 đến 0,71 accuracy trong CV, kể cả khi biết cả 8 tháng. Con số đáng tin ở bài này là chênh lệch giữa các mức làm sạch trong cùng điều kiện, không phải con số tuyệt đối. Không nên suy ra kết quả trên dữ liệu ngân hàng thật.
2. Thí nghiệm che 10% giả định ô thiếu thật xuất hiện ngẫu nhiên. Với `Monthly_Balance` giả định này không đúng: các ô thiếu thuộc nhóm khách khác và có dấu hiệu thiếu không ngẫu nhiên (vẫn chỉ là giả thuyết). Giá trị điền cho `Amount_invested_monthly` và `Monthly_Balance` là ước lượng, không phải số thật.
3. Miền hợp lệ được suy ra từ tần suất, nên có thể loại nhầm vài giá trị thật hiếm nằm sát biên. Kết quả không nhạy với ngưỡng từ 10 đến 100 nhưng cũng không tuyệt đối.
4. Cột `Name` còn lỗi chữ cái (nhiều tên có một chữ thường dính ở cuối, ví dụ `Ashleyh`). Bài không sửa vì cột này không dùng làm đặc trưng.
5. Các tham số làm sạch được tính trên cả 100.000 dòng (không dùng nhãn) trước khi chia train và test. Chỉ có mô hình điền thiếu của hai cột tiền là học xuyên khách, các phần còn lại đều tính trong từng khách. Học lại mô hình đó chỉ trên phần train của từng phần CV không đổi kết quả (macro-F1 chênh −0,01 điểm %, mục 6). Khi dự đoán test, làm sạch chạy trên chuỗi train cộng test (vẫn không dùng nhãn), nên mô hình điền thiếu có nhìn thấy đặc trưng của test.
6. Làm sạch dùng cả các tháng sau của cùng khách để sửa một ô (điền theo tháng gần nhất, trung vị của khách, suy ra lịch sử tín dụng). Đây là sửa lại bản ghi đã có đủ cả kỳ, không phải dự báo. Nếu dùng theo thời gian thực thì chưa có các tháng sau. Riêng đặc trưng đưa vào mô hình chính thì chỉ nhìn tháng hiện tại và các tháng trước.
7. Train và test chỉ chia một lần. Tìm siêu tham số có giới hạn (20 và 8 cấu hình, CV 3 phần).
8. Điểm trên `test.csv` không đo được ở máy vì file không có nhãn. CV ở notebook 4 chia theo khách (khách mới), còn `test.csv` là cùng khách ở các tháng sau, nên hai con số không so trực tiếp được. Mô phỏng chia theo thời gian cho accuracy 0,684 (mục 6), thấp hơn 0,714 của khách giữ kín.

## 8. Chạy lại

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn nbformat nbclient ipykernel
# đặt train.csv và test.csv (tải từ Kaggle) vào thư mục data/
bash run_all.sh
```

`run_all.sh` chạy notebook 1 đến 4 lần lượt, mất khoảng 1 đến 1,5 giờ trên 2 CPU (phần lớn là notebook 4), rồi chạy `make_output.py` (khoảng 2 phút) để tạo `outputs/output.csv`. Cũng có thể mở từng notebook và chạy từ đầu đến cuối (Restart and Run All).

Nếu chỉ cần tạo lại file output (đã có `data/credit_clean.csv`) thì chạy `python3 make_output.py`. Các kiểm tra thêm chạy riêng bằng `python3 check_robustness.py` (khoảng 20 phút, chọn phần cần chạy bằng `--checks`).

Kết quả chạy lại được: notebook 2 cho ra `credit_clean.csv` giống hệt từng byte, `make_output.py` cho ra `output.csv` giống hệt từng byte, và các hạt ngẫu nhiên đều được cố định (ghi trong `data/reports/output_log.json`).

Code làm sạch cũng dùng riêng được như một module:

```python
import sys
sys.path.insert(0, 'src')
import pandas as pd
import credit_cleaning as cc

clean_df = cc.clean(pd.read_csv('data/train.csv', low_memory=False))
assert all(cc.validate_clean(clean_df).values())
```
