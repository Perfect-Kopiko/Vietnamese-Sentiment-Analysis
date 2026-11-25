# Checkpoint mô hình

## Vị trí và dependency

Cài `requirements-demo.txt` từ thư mục gốc. TF-IDF cần scikit-learn 1.7.2, PyVi 0.1.1; BiLSTM cần PyTorch (`torch>=2.0.0`). Thứ tự nhãn chung: **0 = tiêu cực, 1 = trung tính, 2 = tích cực**.

| Checkpoint | Đường dẫn | Sử dụng |
| --- | --- | --- |
| `best_tfidf.pkl` | `checkpoints/best_tfidf.pkl` | TF-IDF + Logistic Regression. |
| `bilstm_no_seg_w2v_best.pt` | `checkpoints/bilstm/bilstm_no_seg_w2v_best.pt` | BiLSTM chính của demo. |
| `bilstm_seg_w2v_weighted_ce_best.pt` | `checkpoints/bilstm/bilstm_seg_w2v_weighted_ce_best.pt` | BiLSTM có tách từ, dùng để đối chiếu. |

## Nạp và kiểm tra

```powershell
Get-FileHash -Path checkpoints/best_tfidf.pkl,checkpoints/bilstm/*.pt -Algorithm SHA256
python -m demo.app
```

TF-IDF lưu pipeline tiền xử lý, phân loại, cấu hình và checksum dữ liệu. BiLSTM nạp vocabulary, cấu hình và embedding từ checkpoint, chạy CPU, không cần Word2Vec gốc. Dùng bản có tách từ bằng `InferencePipeline(bilstm_checkpoint=...)`.

Checkbox chuẩn hóa chỉ đổi đầu vào của lượt dự đoán, không sửa trọng số. Mặc định tắt đúng cấu hình checkpoint chính.

| File | SHA-256 |
| --- | --- |
| `best_tfidf.pkl` | `fb83c59d9bd466001711f7744c119a37cf5b65f161d88ff8da8800156844c704` |
| `bilstm_no_seg_w2v_best.pt` | `af34605c329805d110bf77d20b52ec7b12555e84245e1635560e71f1c11a6674` |
| `bilstm_seg_w2v_weighted_ce_best.pt` | `ae65b3ef4cfb480419b7b01ab5efae54f1121503d11d275dbe87a7ebc43c54b9` |

Checkpoint được loại khỏi Git theo `.gitignore`. Chỉ nạp pickle/joblib và trọng số từ nguồn tin cậy. [Hướng dẫn cài đặt và chạy lại](../README.md).

## PhoBERT đang chờ tích hợp

Đường dẫn dự kiến: `checkpoints/best_phobert/model.pt`. Hiện thiếu checkpoint và mã suy luận; cần kiến trúc, tokenizer/backbone cục bộ, cấu hình tiền xử lý và thứ tự nhãn tương ứng. Demo hiển thị trạng thái chưa sẵn sàng.
