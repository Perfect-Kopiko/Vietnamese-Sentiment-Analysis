# Checkpoint mô hình

## TF-IDF + Logistic Regression

Đặt `best_tfidf.pkl` trong `checkpoints/` và chạy `python -m demo.app`. Cài scikit-learn 1.7.2 và PyVi 0.1.1 theo `requirements-demo.txt`. Chỉ nạp file joblib/pickle từ nguồn tin cậy.

File lưu pipeline tiền xử lý và phân loại, cấu hình, kết quả validation và checksum dữ liệu. Thứ tự nhãn là 0 (tiêu cực), 1 (trung tính), 2 (tích cực).

Kiểm tra SHA-256 bằng `Get-FileHash checkpoints/best_tfidf.pkl` trên PowerShell.

```text
fb83c59d9bd466001711f7744c119a37cf5b65f161d88ff8da8800156844c704
```

Checkpoint nằm ngoài Git theo `.gitignore`. [Hướng dẫn chạy lại và tái lập](../README.md)

## BiLSTM và PhoBERT

| Mô hình | Đường dẫn checkpoint | Cần để chạy trong demo |
| --- | --- | --- |
| BiLSTM | `checkpoints/bilstm/<run_name>_best.pt` | Mã kiến trúc và suy luận khớp `model_state_dict`, `vocab_word2idx` và cấu hình tiền xử lý. |
| PhoBERT | `checkpoints/best_phobert/model.pt` | Tokenizer, cấu hình và mã suy luận nạp backbone từ tài nguyên cục bộ. |

Demo báo chưa sẵn sàng nếu thiếu checkpoint hoặc mã suy luận. Khi tích hợp, đối chiếu thứ tự nhãn và cách tiền xử lý rồi thử đúng checkpoint đã đánh giá trên CPU offline.
