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
| PhoBERT | Tắt | Tiêu cực | 98,4404% | 0,4137% | 1,1459% |
| PhoBERT | Bật | Tiêu cực | 98,5057% | 0,4295% | 1,0648% |

[CSV ba mô hình](demo_three_models.csv): 5 câu mẫu, hai trạng thái chuẩn hóa, ba mô hình; thứ tự xác suất 0/1/2. [CSV TF-IDF và BiLSTM trước tích hợp](demo_predictions.csv) được giữ nguyên. Checkbox đổi đầu vào của checkpoint cố định, khác huấn luyện riêng từng cấu hình trong bảng ablation. Chuẩn hóa bằng từ điển không khôi phục dấu và không đảm bảo cải thiện mọi câu.

## Tích hợp và kiểm thử

TF-IDF, BiLSTM và PhoBERT đã tích hợp; checkpoint BiLSTM chính: `bilstm_no_seg_w2v_best.pt`. Kết quả BiLSTM: [không tách từ](../../results/bilstm_no_seg_w2v.json), [có tách từ](../../results/bilstm_seg_w2v_weighted_ce.json).

PhoBERT dùng checkpoint `best_phobert/model.pt`, tokenizer cục bộ, max length 128, tách từ PyVi. Chuẩn hóa mặc định của checkpoint là bật; trong demo, cờ này luôn theo checkbox chung. Các bảng ablation, mỉa mai, hình và notebook của thành viên PhoBERT được giữ nguyên trong [results](../../results/) và [notebook phân tích lỗi](../../notebooks/02_error_visualization.ipynb).

Đánh giá lại PhoBERT trên toàn bộ 3.166 câu test bằng CPU, không kết nối mạng: Accuracy 93,8092%, Macro-F1 82,5708%, Weighted-F1 93,5620%. Cả ba chỉ số khớp chính xác với `test_metrics.json` đi kèm checkpoint; không huấn luyện lại hoặc ghi đè kết quả của thành viên.

Kiểm thử: **36 test pytest pass**; HTTP 200, 10 yêu cầu cho 30 dự đoán khớp suy luận trực tiếp; đủ ba xác suất, tổng bằng 1 trong sai số; từ chối câu trống/quá dài. App chạy với kết nối mạng ngoài bị chặn; kiểm tra nạp một lần và bật/tắt chuẩn hóa cho cả ba mô hình đều đạt. Qua 20 yêu cầu liên tiếp sau khi khởi động, RAM không tăng dần; mức cao nhất ghi nhận của tiến trình demo khoảng 905 MiB trên môi trường kiểm tra.

Màu chữ và nền của cột kết quả đã có quy tắc riêng cho sáng/tối; các cặp màu kiểm tra có độ tương phản từ 8,2:1. Chưa xác nhận trực quan và thao tác click trên trình duyệt ở bản ba mô hình vì dịch vụ điều khiển trình duyệt không khả dụng.

## Tạo lại bảng số liệu

Từ thư mục gốc, chạy:

```powershell
python -m scripts.prepare_tfidf_report
```

Lệnh đọc kết quả trong `reports/tfidf/`, xuất bảng và Confusion Matrix vào `reports/handoff/tfidf/`, không huấn luyện lại. Xem [hướng dẫn cài đặt, đánh giá, EDA và demo](../../README.md).

## Khai báo sử dụng AI

Có sử dụng OpenAI Codex (ChatGPT) hỗ trợ viết và sửa mã tích hợp demo, soạn tài liệu và kiểm thử. Số liệu báo cáo lấy từ kết quả thực nghiệm và checkpoint của dự án.
