# Vietnamese Sentiment Analysis

Phân loại bình luận sinh viên UIT-VSFC thành ba nhãn 0 (tiêu cực), 1 (trung tính) và 2 (tích cực). Nhánh `feat/tfidf-demo` gồm mô hình TF-IDF + Logistic Regression, thí nghiệm tách từ và nhiễu, notebook EDA và demo Gradio.

## Chạy demo

Chạy từ thư mục gốc dự án bằng Python 3.13.5. Nếu tải mã qua Git, chép `best_tfidf.pkl` từ bản bàn giao vào `checkpoints/`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m demo.app
```

Mở [http://127.0.0.1:7860](http://127.0.0.1:7860), nhập bình luận hoặc chọn câu mẫu, rồi bấm **So sánh mô hình** để xem nhãn và xác suất. Demo dùng checkpoint cục bộ và chạy offline sau khi cài thư viện. BiLSTM và PhoBERT báo chưa sẵn sàng khi thiếu checkpoint hoặc mã suy luận. [Hướng dẫn checkpoint](checkpoints/README.md)

Đổi cổng bằng `$env:GRADIO_SERVER_PORT = "7861"`. Trên Linux/macOS, dùng `./.venv/bin/python` thay cho đường dẫn Python trong các lệnh.

## Dữ liệu và kiểm chứng

Cài thêm thư viện để chạy EDA, huấn luyện và kiểm thử. `requirements.txt` là danh sách thư viện chung của nhóm.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-tfidf.txt
.\.venv\Scripts\python.exe -m pip check
```

Bản bàn giao có sẵn ba file CSV trong `data/raw/`. Nếu tải mã qua Git, dùng các lệnh dưới đây để tải đúng phiên bản UIT-VSFC và tạo `data/raw/manifest.json` chứa checksum. Script báo lỗi nếu CSV hiện có khác dữ liệu nguồn.

```powershell
$env:HF_HOME = Join-Path (Get-Location) ".cache\hf"
.\.venv\Scripts\python.exe data/download_data.py
```

| Split | Tiêu cực | Trung tính | Tích cực | Tổng |
| --- | ---: | ---: | ---: | ---: |
| Train | 5.325 | 458 | 5.643 | 11.426 |
| Validation | 705 | 73 | 805 | 1.583 |
| Test | 1.409 | 167 | 1.590 | 3.166 |

Dữ liệu gốc có trên Hugging Face. [UIT NLP / Vietnamese Students’ Feedback Corpus](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)

Kiểm tra checkpoint có sẵn bằng các lệnh dưới đây. Chạy lệnh cuối trong terminal thứ hai khi demo đang mở.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.evaluate_tfidf --output-dir .cache/baseline-check
.\.venv\Scripts\python.exe -m scripts.check_demo --url http://127.0.0.1:7860
```

Script đánh giá đối chiếu checksum dữ liệu với checkpoint. Kết quả kiểm tra demo nằm trong `.cache/demo_check/`.

## Kết quả baseline

`configs/tfidf_lr.yaml` dùng TF-IDF unigram/bigram và Logistic Regression cân bằng lớp. Có 8 tổ hợp bật/tắt PyVi, chuẩn hóa teencode và bỏ dấu, mỗi tổ hợp thử `C=1` và `C=4`. Pipeline học trên train và được chọn theo Macro-F1 trên validation sạch trước khi đánh giá test.

Checkpoint dùng PyVi, giữ dấu, tắt chuẩn hóa teencode và chọn `C=4`. Accuracy / Macro-F1 đạt 90,02% / 76,87% trên validation sạch và 88,12% / 72,94% trên test sạch.

| Thay đổi trên test | Trước (Accuracy / Macro-F1) | Sau (Accuracy / Macro-F1) |
| --- | ---: | ---: |
| Bật tách từ trên câu sạch | 88,19% / 73,04% | 88,12% / 72,94% |
| Bỏ dấu ở train và suy luận, đánh giá trên test không dấu | 38,50% / 36,48% | 81,74% / 67,38% |
| Bật chuẩn hóa teencode, đánh giá trên test nhiễu teencode | 86,86% / 71,59% | 87,93% / 72,95% |

Mỗi hàng chỉ đổi một tùy chọn. Nhiễu teencode dùng seed cố định và cùng từ điển với bước chuẩn hóa. [Bảng 8 cấu hình](reports/tfidf/test_comparison.csv) · [Kết quả chi tiết](reports/tfidf/experiment_results.csv) · [Phiên bản và checksum](reports/tfidf/run_metadata.json)

Để chạy lại thí nghiệm, lưu checkpoint và báo cáo vào thư mục riêng.

```powershell
.\.venv\Scripts\python.exe -m scripts.train_tfidf --output-dir .cache/tfidf-replay --checkpoint .cache/tfidf-replay/best_tfidf.pkl
```

Nếu chưa có checkpoint, `python -m scripts.train_tfidf` lưu `checkpoints/best_tfidf.pkl` và ghi báo cáo vào `reports/tfidf/`.

## Notebook EDA

Mở notebook và chọn kernel `.venv`, hoặc chạy lệnh sau. [Notebook EDA](notebooks/01_eda_data_stats.ipynb)

```powershell
.\.venv\Scripts\python.exe -m scripts.execute_eda --repeats 1
```

Notebook lưu các bảng và biểu đồ PNG (300 dpi), PDF trong `reports/eda/`. [Phân bố nhãn](reports/eda/label_distribution.png) · [Độ dài bình luận](reports/eda/comment_lengths.pdf) · [Độ dài theo cảm xúc](reports/eda/length_by_sentiment.pdf)
