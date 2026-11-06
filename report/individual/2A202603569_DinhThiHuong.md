<img width="1337" height="296" alt="{0428602A-26AE-4EA7-8A9F-ACAFF7DC75F0}" src="https://github.com/user-attachments/assets/0c051b45-ab3d-4845-8330-0e6a88557b70" /># Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Mô hình | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
|---|---|---|---|---|
| BoW + LogReg | 0.901 | 0.857 | 0.129 | 0.000 |
| BoW + MLP | 0.942 | 0.889 | 0.204 | 0.000 |
| Encoder + kNN | 0.928 | 0.950 | 65.407 | 0.000 |
| Encoder + LogReg | 0.954 | 0.919 | 33.274 | 0.000 |
| Encoder + MLP head | 0.949 | 0.919 | 34.183 | 0.000 |
| LLM nhỏ zero-shot | — | 0.947 | 1105.517 | 0.010 |
| LLM lớn zero-shot | — | 0.919 | 1207.498 | 0.102 |
| LLM nhỏ few-shot | — | 0.974 | 986.413 | 0.023 |
## 3. Dự đoán và thực tế
### 1. Learning rate lớn — `ml_lr`

- **Dự đoán:** Với `lr=50`, loss dao động mạnh hoặc tăng vọt, mô hình khó hội tụ.
- **Thực tế:** Loss tăng vọt khoảng 2.6 lúc đầu, sau đó giảm nhanh xuống 0.006 sau 300 epoch.
- **Giải thích:** Dự đoán đã đánh giá quá chắc tác động của learning rate lớn. Khả năng hội tụ còn phụ thuộc dữ liệu và đặc trưng; trong thí nghiệm này, mô hình vẫn tối ưu được train loss. Tuy nhiên, train loss thấp chưa chứng minh F1 test tốt hơn.

### 2. Model lớn so với model nhỏ — `llm_models`

- **Dự đoán:** Model lớn có F1 cao hơn model nhỏ khoảng 0.02–0.05.
- **Thực tế:** Model lớn đạt F1 0.919, thấp hơn model nhỏ đạt 0.947, trong khi chi phí ước tính cao hơn khoảng 10 lần.
- **Giải thích:** Model lớn không đảm bảo tốt hơn ở mọi bài toán. SUB chỉ có 40 tin và hai model chênh nhau một tin đúng, nên kết quả còn thiếu chắc chắn.

### 3. Khả năng xử lý case khó — `hard_eval`

- **Dự đoán:** LLM nhỏ zero-shot đạt accuracy khoảng 85%, còn prompt phòng thủ khoảng 90%; prompt injection có thể là nhóm khó.
- **Thực tế:** Trên bộ 55 case gồm 50 case gốc và 5 case bổ sung, cả hai phương án LLM đều đạt accuracy 100%, kể cả nhóm prompt injection.
- **Giải thích:** Bộ case này chưa đủ khó để phân biệt hai phương án LLM. Prompt zero-shot hiện tại cũng đã yêu cầu coi tin nhắn là dữ liệu và không làm theo chỉ dẫn bên trong. Kết quả 100% trên bộ nhỏ, viết tay không đảm bảo chất lượng tương tự ngoài thực tế.
## 4. Phân tích 50 case khó
Tôi đánh giá trên **55 case**, gồm 50 case gốc và 5 case bổ sung.

**BoW + LogReg tụt nhiều nhất:** F1 giảm từ **0.857 trên SUB xuống 0.263 trên bộ khó**, giảm 0.594; accuracy chỉ đạt **49.1%**. Nhóm khó nhất là **spam giọng tự nhiên**, mô hình đúng 0%. Nhóm né từ khóa cũng chỉ đạt khoảng 14%.

Hai case cụ thể mà BoW + LogReg đoán nhầm thành ham:

1. **Case #14:** “Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back”.
   - Nhãn thật là spam: lời mời kiếm tiền dễ dàng nhằm dụ người nhận phản hồi.
   - Tin viết như lời chia sẻ cá nhân. BoW chỉ ghi nhận các từ riêng lẻ nên khó nhận ra ý đồ lừa việc làm qua toàn bộ câu.

2. **Case #18:** “Mate you have to see this, I doubled my savings in 2 weeks with this app, use my code and we both get a bonus”.
   - Nhãn thật là spam: quảng bá ứng dụng đầu tư bằng lời hứa lợi nhuận và mã giới thiệu.
   - Giọng nói thân mật giống tin bạn bè, trong khi dấu hiệu đáng ngờ nằm ở sự kết hợp giữa lời hứa tăng tiền nhanh và lời mời sử dụng ứng dụng. BoW không biểu diễn đầy đủ mối liên hệ này.

Các phương án Embedding cũng gặp khó nhất với spam giọng tự nhiên, chỉ đúng khoảng **18–36%**. Hai phương án LLM đều đúng **55/55 case**, nhưng bộ kiểm tra nhỏ và viết tay chưa đủ để kết luận chúng đạt độ chính xác tuyệt đối ngoài thực tế.
## 5. Thử thách cá nhân

**Đề #2:** Giảm vocab còn 100 từ và tăng lên 5000 từ, train lại Logistic Regression, báo F1 test và giải thích xu hướng.

**Cách làm:** Tạo vocab từ tập train với ba kích thước 100, 2000 và 5000 từ. Với mỗi vocab, vector hóa lại train/test, huấn luyện với cùng `lr=0.5`, `epochs=300` và ngưỡng phân loại 0.5 để so sánh.

| Kích thước vocab | Precision | Recall | F1 test |
|---|---|---|---|
| 100 | 0.9737 | 0.7475 | 0.8457 |
| 2000 | 0.9880 | 0.8283 | 0.9011 |
| 5000 | 0.9880 | 0.8283 | 0.9011 |

**Giải thích:** Vocab 100 từ bỏ sót nhiều dấu hiệu nhận diện spam nên recall và F1 thấp hơn. Tăng lên 2000 từ giúp cải thiện kết quả, nhưng tăng tiếp lên 5000 từ không cải thiện các chỉ số trong lần thử này. Tôi chọn 2000 từ vì đạt chất lượng tương đương 5000 từ với ít đặc trưng hơn.

## 6. Phương án cho sản phẩm thật

Tôi chọn **BoW + MLP làm mô hình xử lý chính ban đầu**. Mô hình đạt F1 test **0.942**, p95 **0.204 ms/tin** và throughput khoảng **46.498 tin/giây** trong phép đo hiện tại, đáp ứng yêu cầu độ trễ và vượt mức trung bình khoảng 116 tin/giây. Tuy nhiên, cần kiểm thử tải thực tế, đặc biệt khi lưu lượng tăng đột biến.

Đánh đổi là mô hình còn yếu trước các kiểu spam mới: accuracy trên bộ 55 case khó chỉ đạt **60%**. Encoder + LogReg có F1 test cao hơn, **0.954**, và p95 **33.274 ms**, nhưng throughput hiện tại chỉ khoảng **33 tin/giây**, cần mở rộng tài nguyên nếu dùng cho toàn bộ lưu lượng.

Tôi đề xuất router dùng BoW + MLP cho các tin có dự đoán rõ ràng, chuyển các tin lưỡng lự sang Encoder + LogReg. LLM hỗ trợ kiểm tra bổ sung bất đồng bộ vì p95 gần một giây, vượt SLA. Cần chọn ngưỡng router trên tập validation và đo lại chất lượng, công suất, độ trễ của toàn hệ thống.

Nếu gọi LLM cho toàn bộ 300 triệu SMS/tháng (30 ngày), chi phí API ước tính từ bảng đã làm tròn là khoảng **3000 USD/tháng** với model nhỏ zero-shot hoặc **6900 USD/tháng** với model nhỏ few-shot. Chi phí này chưa gồm hạ tầng; mô hình local cũng có chi phí máy chạy dù bảng ghi phí API bằng 0.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Cấu hình API | Đã thêm key nhưng notebook chưa nhận, sửa thế nào? | Hướng dẫn đọc Colab Secrets và gán biến môi trường | Chạy kiểm tra, xác nhận đã nhận key |
| Phần 4 — Prompt và parse | Hoàn thành hàm tạo prompt và đọc nhãn | Viết prompt zero-shot/few-shot và parse_label | Thử hai tin, nhận nhãn [1, 0] và invalid=0 |
| Phần 4.A — Zero-shot | Viết đầy đủ code so sánh hai model | Thêm khoảng nghỉ, xử lý rate limit và đặt giới hạn 30 token | Model nhỏ F1=0.947, model lớn F1=0.919, invalid=0 |
| Phần 4.B — Few-shot | Viết code chạy few-shot và so với zero-shot | Thêm bốn ví dụ từ train, tính F1 và chi phí | F1=0.974, chi phí tăng 2.25 lần, invalid=0 |
| Stress test | Viết code đánh giá các model trên case khó | Viết vòng lặp đánh giá và bảng kết quả theo nhóm | Chạy và đối chiếu bảng kết quả |
| Bonus | Đề xuất năm case bổ sung | Viết năm tin spam/ham khó | Thêm thành 55 case; hai phương án LLM đạt accuracy 100% |

**Điều đã học:** Tôi hiểu rằng accuracy cao chưa chắc phát hiện spam tốt, vì mô hình luôn đoán ham cũng đạt khoảng 87.37% accuracy nhưng recall spam bằng 0. Model lớn hơn hoặc vocab lớn hơn không đảm bảo kết quả tốt hơn, nên cần kiểm chứng thay vì suy đoán. Khi chọn mô hình cho sản phẩm, tôi cần xem đồng thời chất lượng, độ trễ, công suất và chi phí, đồng thời kiểm tra các tình huống khác dữ liệu huấn luyện.
