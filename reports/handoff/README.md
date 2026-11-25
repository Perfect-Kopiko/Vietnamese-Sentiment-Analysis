# Kết quả TF-IDF và demo

## TF-IDF

Checkpoint dùng PyVi, giữ dấu, không chuẩn hóa teencode khi huấn luyện, `C=4`; chọn theo Macro-F1 trên validation sạch. Kết quả trên 3.166 mẫu test sạch:

| Accuracy | Macro-F1 | F1 Neg | F1 Neu | F1 Pos |
| ---: | ---: | ---: | ---: | ---: |
| 88,12% | 72,94% | 90,52% | 36,57% | 91,72% |

[Mọi cấu hình](tfidf/metrics_all.csv) · [Checkpoint demo](tfidf/metrics_selected.csv): Accuracy, Macro-F1 và F1 từng lớp trên validation/test sạch, không dấu và teencode.

Confusion Matrix test: [sạch](tfidf/confusion_matrix_clean.png), [không dấu](tfidf/confusion_matrix_no_accents.png), [teencode](tfidf/confusion_matrix_teencode.png). CSV cùng tên lưu số đếm; hàng = nhãn thật, cột = dự đoán, thứ tự Neg/Neu/Pos.

## EDA

[Phân bố nhãn](../eda/label_distribution.png) · [Độ dài bình luận](../eda/comment_lengths.png) · [Độ dài theo nhãn](../eda/length_by_sentiment.png). Ảnh PNG 300 dpi, bản PDF và bảng CSV nằm trong `reports/eda/`.

## Demo bật và tắt teencode

Câu nhập: `gv dạy ko rõ ràng, sv k hiểu bài`.

| Mô hình | Chuẩn hóa | Nhãn | P Neg | P Neu | P Pos |
| --- | --- | --- | ---: | ---: | ---: |
| TF-IDF | Tắt | Tích cực | 2,5587% | 0,1118% | 97,3295% |
| TF-IDF | Bật | Tiêu cực | 96,2811% | 0,5167% | 3,2023% |
| BiLSTM | Tắt | Tích cực | 15,3151% | 3,3897% | 81,2952% |
| BiLSTM | Bật | Tiêu cực | 99,6417% | 0,0296% | 0,3286% |

[CSV demo](demo_predictions.csv): 5 câu mẫu, hai trạng thái chuẩn hóa, hai mô hình; thứ tự xác suất 0/1/2. Checkbox đổi đầu vào của checkpoint cố định, khác huấn luyện riêng từng cấu hình trong bảng ablation. Chuẩn hóa bằng từ điển không khôi phục dấu và không đảm bảo cải thiện mọi câu.

## Tích hợp và kiểm thử

TF-IDF và BiLSTM đã tích hợp; checkpoint chính: `bilstm_no_seg_w2v_best.pt`. Kết quả BiLSTM: [không tách từ](../../results/bilstm_no_seg_w2v.json), [có tách từ](../../results/bilstm_seg_w2v_weighted_ce.json).

**PhoBERT đang chờ tích hợp**: thiếu checkpoint và mã suy luận, tokenizer/backbone cục bộ cùng cấu hình tương ứng.

Kiểm thử: **30 test pytest pass**; HTTP 200, 10 yêu cầu cho 20 dự đoán khớp suy luận trực tiếp; đủ ba xác suất, tổng bằng 1 trong sai số; từ chối câu trống/quá dài. Ảnh giao diện sáng/tối được kiểm tra bằng Chrome headless. Chưa kiểm thử thao tác click trên trình duyệt và suy luận đủ ba mô hình.

## Tạo lại bảng số liệu

Từ thư mục gốc, chạy:

```powershell
python -m scripts.prepare_tfidf_report
```

Lệnh đọc kết quả trong `reports/tfidf/`, xuất bảng và Confusion Matrix vào `reports/handoff/tfidf/`, không huấn luyện lại. Xem [hướng dẫn cài đặt, đánh giá, EDA và demo](../../README.md).

## Khai báo sử dụng AI

Có sử dụng OpenAI Codex (ChatGPT) hỗ trợ viết và sửa mã tích hợp demo, soạn tài liệu và kiểm thử. Số liệu báo cáo lấy từ kết quả thực nghiệm và checkpoint của dự án.
