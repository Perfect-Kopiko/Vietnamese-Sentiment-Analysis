"""
Script tải và lưu trữ bộ dữ liệu UIT-VSFC từ HuggingFace datasets.
Chuẩn hóa các cột thành 'text' và 'label', lưu ra thư mục data/raw/.
"""

import os
from typing import Dict
import pandas as pd
from datasets import load_dataset


DATASET_NAME = "uit-nlp/vietnamese_students_feedback"
LABEL_MAPPING: Dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def download_and_save_data(output_dir: str = "data/raw") -> None:
    """
    Tải bộ dữ liệu UIT-VSFC từ HuggingFace datasets,
    chuẩn hóa tên cột sentence -> text, sentiment -> label,
    và lưu thành các tệp train.csv, validation.csv, test.csv.

    Args:
        output_dir: Đường dẫn thư mục lưu file kết quả.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Đang tải bộ dữ liệu '{DATASET_NAME}' từ HuggingFace...")

    try:
        dataset = load_dataset(DATASET_NAME, revision="refs/convert/parquet")
    except Exception as exc:
        print(f"[!] Lỗi khi tải bộ dữ liệu: {exc}")
        raise

    splits = ["train", "validation", "test"]

    for split in splits:
        if split not in dataset:
            print(f"[!] Cảnh báo: Split '{split}' không tồn tại trong dataset.")
            continue

        split_ds = dataset[split]
        df = pd.DataFrame(split_ds)

        # Đổi tên cột
        column_rename = {}
        if "sentence" in df.columns:
            column_rename["sentence"] = "text"
        if "sentiment" in df.columns:
            column_rename["sentiment"] = "label"

        df = df.rename(columns=column_rename)

        # Giữ lại các cột quan trọng
        if "text" in df.columns and "label" in df.columns:
            df = df[["text", "label"]]
        
        # Xử lý các dòng null/nan nếu có
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(int)

        file_path = os.path.join(output_dir, f"{split}.csv")
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"[+] Đã lưu '{split}' vào: {file_path}")

        # Thống kê số lượng và phân bố nhãn
        print(f"--- Thống kê split: {split.upper()} ---")
        print(f"Tổng số mẫu: {len(df):,}")
        label_counts = df["label"].value_counts().sort_index()
        for lbl_val, count in label_counts.items():
            lbl_name = LABEL_MAPPING.get(lbl_val, f"Unknown({lbl_val})")
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  Nhãn {lbl_val} ({lbl_name}): {count:,} ({pct:.2f}%)")
        print()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    raw_dir = os.path.join(project_root, "data", "raw")
    download_and_save_data(output_dir=raw_dir)
