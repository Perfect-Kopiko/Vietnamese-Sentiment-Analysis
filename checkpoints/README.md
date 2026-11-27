# Checkpoint mô hình

## Vị trí và dependency

Cài `requirements-demo.txt` từ thư mục gốc. TF-IDF dùng scikit-learn 1.7.2, PyVi 0.1.1; BiLSTM và PhoBERT dùng PyTorch (`torch>=2.6.0`). PhoBERT dùng Transformers 5.24.0. Thứ tự nhãn chung: **0 = tiêu cực, 1 = trung tính, 2 = tích cực**.

| Checkpoint | Đường dẫn | Sử dụng |
| --- | --- | --- |
| `best_tfidf.pkl` | `checkpoints/best_tfidf.pkl` | TF-IDF + Logistic Regression. |
| `bilstm_no_seg_w2v_best.pt` | `checkpoints/bilstm/bilstm_no_seg_w2v_best.pt` | BiLSTM chính của demo. |
| `bilstm_seg_w2v_weighted_ce_best.pt` | `checkpoints/bilstm/bilstm_seg_w2v_weighted_ce_best.pt` | BiLSTM có tách từ, dùng để đối chiếu. |
| `model.pt` và tokenizer | `checkpoints/best_phobert/` | PhoBERT-base-v2 đã fine-tune, ba lớp. |

## Nạp và kiểm tra

Trước khi chạy toàn bộ integration test (`python -m pytest -q`), chuẩn bị đủ bốn checkpoint trong bảng, tokenizer PhoBERT và dữ liệu `data/raw/` theo [README gốc](../README.md). Bộ test đối chiếu cả hai checkpoint BiLSTM với pipeline huấn luyện; test bị bỏ qua vì thiếu checkpoint chưa xác nhận được mô hình tương ứng.

```powershell
Get-FileHash -Path checkpoints/best_tfidf.pkl,checkpoints/bilstm/*.pt,checkpoints/best_phobert/model.pt -Algorithm SHA256
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
python -m demo.app
```

TF-IDF lưu pipeline tiền xử lý, phân loại, cấu hình và checksum dữ liệu. BiLSTM nạp vocabulary, cấu hình và embedding từ checkpoint, chạy CPU, không cần Word2Vec gốc. Dùng bản có tách từ bằng `InferencePipeline(bilstm_checkpoint=...)`.

Checkbox chuẩn hóa mặc định tắt, phù hợp với TF-IDF và BiLSTM chính. Với PhoBERT, bật checkbox để dùng đúng cấu hình huấn luyện.

| File | SHA-256 |
| --- | --- |
| `best_tfidf.pkl` | `fb83c59d9bd466001711f7744c119a37cf5b65f161d88ff8da8800156844c704` |
| `bilstm_no_seg_w2v_best.pt` | `af34605c329805d110bf77d20b52ec7b12555e84245e1635560e71f1c11a6674` |
| `bilstm_seg_w2v_weighted_ce_best.pt` | `ae65b3ef4cfb480419b7b01ab5efae54f1121503d11d275dbe87a7ebc43c54b9` |
| `best_phobert/model.pt` | `12b9fa0e8bfad82c799bd851d37d7ca8b99239886ba70f88808c77f7a8678aed` |

Checkpoint được loại khỏi Git theo `.gitignore`. Chỉ nạp pickle/joblib và trọng số từ nguồn tin cậy. [Hướng dẫn cài đặt và chạy lại](../README.md).

## Tài nguyên PhoBERT offline

Giải nén `best_phobert-20261126T060206Z-1-001.zip` vào `checkpoints/`. ZIP chứa sẵn thư mục `best_phobert/`: giữ nguyên `model.pt`, `config.json`, `tokenizer_config.json`, `added_tokens.json`, `vocab.txt`, `bpe.codes`, `best_val_metrics.json` và `test_metrics.json`. Không giải nén ZIP mã nguồn đè lên repo.

`config.json` là cấu hình huấn luyện, không phải cấu hình RoBERTa của Hugging Face. [Cấu hình backbone chính thức](https://huggingface.co/vinai/phobert-base-v2/blob/main/config.json) được lưu sẵn ở `configs/phobert_backbone.json` (SHA-256: `26d21a304fdea8fada88e7c6b7c7fef15c902c690bfb0fa47f60bdb61c68307a`). Không cần tải thêm bộ trọng số `phobert-base-v2`.

`demo/phobert_inference.py` nạp đầy đủ `state_dict` vào `PhoBERTClassifier` trên CPU và chạy ở chế độ suy luận. Tokenizer BPE chỉ đọc file cục bộ, max length = 128, có tách từ PyVi. Khi gọi riêng `PhoBERTInference.predict_proba(text)` không truyền cờ, chuẩn hóa mặc định theo checkpoint; demo luôn truyền trạng thái checkbox.

PhoBERT dùng memory map để giảm bộ nhớ tạm lúc nạp. Thư mục checkpoint và ZIP trọng số được loại khỏi Git. Cần chuẩn bị môi trường Python và dependency trước khi chạy offline.
