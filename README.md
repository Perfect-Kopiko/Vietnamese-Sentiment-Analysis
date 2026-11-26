# Vietnamese Sentiment Analysis

Phân loại bình luận sinh viên UIT-VSFC thành ba nhãn: 0 (tiêu cực), 1 (trung tính), 2 (tích cực). Demo so sánh TF-IDF + Logistic Regression, BiLSTM và PhoBERT trên cùng một bình luận.

## Cài đặt

Chạy từ thư mục gốc với Python 3.13.5:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tfidf.txt
.\.venv\Scripts\python.exe -m pip check
```

Chỉ chạy demo: cài `requirements-demo.txt`. `requirements.txt` chứa thư viện chung để huấn luyện và phân tích của nhóm; các phiên bản dùng chung tương thích với demo. Môi trường đã kiểm tra: Python 3.13.5, PyTorch 2.14.0, Transformers 5.24.0. Trên Linux/macOS, dùng `./.venv/bin/python` thay `.\.venv\Scripts\python.exe`.

## Dữ liệu

Nguồn: [UIT NLP / Vietnamese Students’ Feedback Corpus](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback).

Bản bàn giao có ba CSV trong `data/raw/`. Nếu thiếu, chạy lệnh sau để tải đúng phiên bản và tạo `data/raw/manifest.json` chứa checksum. Script từ chối ghi đè CSV khác nguồn.

```powershell
$env:HF_HOME = Join-Path (Get-Location) ".cache\hf"
.\.venv\Scripts\python.exe data/download_data.py
```

| Split | Tiêu cực | Trung tính | Tích cực | Tổng |
| --- | ---: | ---: | ---: | ---: |
| Train | 5.325 | 458 | 5.643 | 11.426 |
| Validation | 705 | 73 | 805 | 1.583 |
| Test | 1.409 | 167 | 1.590 | 3.166 |

## Checkpoint và demo

Đặt các file theo cấu trúc:

```text
checkpoints/
├── best_tfidf.pkl
├── bilstm/
│   ├── bilstm_no_seg_w2v_best.pt
│   └── bilstm_seg_w2v_weighted_ce_best.pt
└── best_phobert/
    ├── model.pt
    ├── config.json
    ├── tokenizer_config.json
    ├── added_tokens.json
    ├── vocab.txt
    └── bpe.codes
```

Checkpoint BiLSTM chính là `bilstm_no_seg_w2v_best.pt`; bản có tách từ dùng để đối chiếu. Giải nén ZIP trọng số PhoBERT vào `checkpoints/` để có đúng thư mục `best_phobert/` như trên. Giữ nguyên `config.json` huấn luyện trong ZIP; cấu hình backbone nằm riêng tại `configs/phobert_backbone.json`. Xem [hướng dẫn checkpoint](checkpoints/README.md).

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m demo.app
```

Mở [http://127.0.0.1:7860](http://127.0.0.1:7860), nhập bình luận rồi bấm **So sánh mô hình**. Ba cột hiển thị nhãn và xác suất cả ba lớp. Checkbox **Bật chuẩn hóa Teencode / Nhiễu** mặc định tắt, áp dụng cho cả ba mô hình.

Demo nạp mỗi mô hình một lần khi khởi động. BiLSTM và PhoBERT chạy CPU ở chế độ suy luận, không cần Word2Vec gốc hay tải trọng số nền từ Hugging Face. PhoBERT dùng tokenizer BPE trong checkpoint, tách từ bằng PyVi và cắt/pad đến 128 token. Checkbox chỉ thay đổi chuẩn hóa nhiễu của lượt dự đoán; không đổi tách từ, nhãn hay trọng số. Checkpoint PhoBERT được huấn luyện với chuẩn hóa bật, nên muốn dùng đúng cấu hình huấn luyện hãy bật checkbox.

Cần cài dependency trước khi ngắt mạng. Khi các checkpoint và tài nguyên trên đã có, demo chạy offline; không cần dataset để suy luận và không huấn luyện lại.

Đổi cổng bằng `$env:GRADIO_SERVER_PORT = "7861"`. Có thể chạy trực tiếp `python demo/app.py`.

## Huấn luyện và đánh giá

`configs/tfidf_lr.yaml` dùng TF-IDF unigram/bigram và Logistic Regression cân bằng lớp. Có 8 tổ hợp bật/tắt PyVi, chuẩn hóa teencode và bỏ dấu, mỗi tổ hợp thử `C=1`, `C=4`. Mô hình học trên train, chọn theo Macro-F1 validation sạch rồi đánh giá test.

Chạy lại thí nghiệm với thư mục riêng để giữ kết quả cũ:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_tfidf --output-dir .cache/tfidf-replay --checkpoint .cache/tfidf-replay/best_tfidf.pkl
```

Lệnh `python -m scripts.train_tfidf` mặc định ghi vào `checkpoints/best_tfidf.pkl` và `reports/tfidf/`. Xem tùy chọn bằng `python scripts/train_tfidf.py --help`.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.evaluate_tfidf --output-dir .cache/baseline-check
```

Script đánh giá kiểm tra checksum dữ liệu với checkpoint. Khi demo đang chạy, mở terminal khác để kiểm tra API và checkbox:

```powershell
.\.venv\Scripts\python.exe -m scripts.check_demo --url http://127.0.0.1:7860
```

Kết quả kiểm tra demo nằm trong `.cache/demo_check/`.

## Kết quả TF-IDF

Checkpoint dùng PyVi, giữ dấu, tắt chuẩn hóa teencode và chọn `C=4`. Accuracy / Macro-F1 đạt 90,02% / 76,87% trên validation sạch và 88,12% / 72,94% trên test sạch.

| Thay đổi trên test | Trước (Accuracy / Macro-F1) | Sau (Accuracy / Macro-F1) |
| --- | ---: | ---: |
| Bật tách từ trên câu sạch | 88,19% / 73,04% | 88,12% / 72,94% |
| Bỏ dấu ở train và suy luận, đánh giá trên test không dấu | 38,50% / 36,48% | 81,74% / 67,38% |
| Bật chuẩn hóa teencode, đánh giá trên test nhiễu teencode | 86,86% / 71,59% | 87,93% / 72,95% |

Mỗi hàng chỉ đổi một tùy chọn. Nhiễu teencode dùng seed cố định và cùng từ điển với bước chuẩn hóa. [Bảng 8 cấu hình](reports/tfidf/test_comparison.csv) · [Kết quả chi tiết](reports/tfidf/experiment_results.csv) · [Phiên bản và checksum](reports/tfidf/run_metadata.json)

## EDA và báo cáo

Mở [notebook EDA](notebooks/01_eda_data_stats.ipynb) với kernel `.venv`, hoặc chạy:

```powershell
.\.venv\Scripts\python.exe -m scripts.execute_eda --repeats 1
```

Bảng CSV, ảnh PNG 300 dpi và PDF được lưu trong `reports/eda/`: [phân bố nhãn](reports/eda/label_distribution.png), [độ dài bình luận](reports/eda/comment_lengths.pdf), [độ dài theo cảm xúc](reports/eda/length_by_sentiment.pdf).

[Báo cáo bàn giao](reports/handoff/README.md) gồm F1 từng lớp, Confusion Matrix, kết quả demo, hướng dẫn tạo lại bảng và khai báo sử dụng AI.
