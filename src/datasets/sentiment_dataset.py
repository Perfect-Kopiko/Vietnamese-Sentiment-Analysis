"""
PyTorch Dataset và DataLoader cho bài toán Phân tích cảm xúc văn bản tiếng Việt.
Tích hợp quy trình tiền xử lý, mã hóa Vocabulary, padding/truncation và attention mask.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Đảm bảo đường dẫn gốc của project có trong sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.datasets.vocab import Vocabulary
from src.preprocessing import preprocess_pipeline


class VietnameseSentimentDataset(Dataset):
    """
    Dataset cho bài toán phân loại cảm xúc tiếng Việt (3 lớp: 0=Negative, 1=Neutral, 2=Positive).
    Mỗi mẫu được tiền xử lý qua preprocess_pipeline, mã hóa bằng Vocabulary,
    sau đó padding/truncate về max_len cố định.
    """

    def __init__(
        self,
        csv_path: str,
        vocab: Vocabulary,
        max_len: int = 64,
        use_word_segment: bool = True,
        normalize_noise: bool = False
    ):
        """
        Khởi tạo VietnameseSentimentDataset.

        Args:
            csv_path: Đường dẫn tệp CSV dữ liệu (chứa các cột 'text' và 'label').
            vocab: Đối tượng Vocabulary đã được build.
            max_len: Chiều dài tối đa của câu sau khi padding/cắt.
            use_word_segment: Có kích hoạt tách từ ghép tiếng Việt (pyvi) hay không.
            normalize_noise: Có chuẩn hóa teencode/từ lóng về từ chuẩn hay không.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu tại: {csv_path}")

        self.csv_path = csv_path
        self.vocab = vocab
        self.max_len = max_len
        self.use_word_segment = use_word_segment
        self.normalize_noise = normalize_noise

        df = pd.read_csv(csv_path)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError(f"Tệp CSV '{csv_path}' phải chứa 2 cột 'text' và 'label'.")

        # Xử lý các dòng null/nan nếu có
        df["text"] = df["text"].fillna("").astype(str)
        df["label"] = df["label"].astype(int)

        self.texts: List[str] = df["text"].tolist()
        self.labels: List[int] = df["label"].tolist()

        # Thực hiện tiền xử lý toàn bộ tập văn bản
        print(f"[*] Đang tiền xử lý {len(self.texts):,} mẫu từ '{os.path.basename(csv_path)}'...")
        self.processed_texts: List[str] = [
            preprocess_pipeline(
                t,
                use_segment=self.use_word_segment,
                normalize_noise=self.normalize_noise
            )
            for t in self.texts
        ]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Lấy một mẫu dữ liệu tại chỉ số idx.

        Returns:
            dict: {
                'input_ids': torch.LongTensor shape (max_len,),
                'attention_mask': torch.LongTensor shape (max_len,),
                'label': torch.LongTensor scalar (giá trị 0, 1, hoặc 2)
            }
        """
        processed_text = self.processed_texts[idx]
        token_ids = self.vocab.encode(processed_text)

        # Cắt nếu dài hơn max_len
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]

        seq_len = len(token_ids)
        # Pad token 0 (<PAD>) ở cuối nếu ngắn hơn max_len
        pad_len = self.max_len - seq_len
        input_ids = token_ids + [Vocabulary.PAD_IDX] * pad_len
        # Attention mask: 1 cho token thật, 0 cho token <PAD>
        attention_mask = [1] * seq_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def create_dataloaders(
    train_path: str = "data/raw/train.csv",
    val_path: str = "data/raw/validation.csv",
    test_path: str = "data/raw/test.csv",
    batch_size: int = 64,
    max_len: int = 64,
    min_freq: int = 2,
    use_word_segment: bool = True,
    normalize_noise: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
    """
    Tiện ích xây dựng Vocabulary trên tập train và trả về các DataLoader cho 3 tập train, val, test.

    Quy tắc quan trọng:
    - Vocabulary CHỈ được fit trên tập train nhằm tránh data leakage (rò rỉ dữ liệu).
    - Tập val và test sử dụng Vocabulary của tập train để encode và thống kê OOV rate.

    Args:
        train_path: Đường dẫn file train.csv.
        val_path: Đường dẫn file validation.csv.
        test_path: Đường dẫn file test.csv.
        batch_size: Kích thước batch.
        max_len: Chiều dài tối đa của câu.
        min_freq: Tần suất từ tối thiểu để đưa vào vocab.
        use_word_segment: Có kích hoạt tách từ tiếng Việt hay không.
        normalize_noise: Có chuẩn hóa teencode hay không.
        num_workers: Số luồng nạp dữ liệu cho PyTorch DataLoader.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Vocabulary]:
            train_loader, val_loader, test_loader, vocab
    """
    print("=" * 60)
    print("[*] KHỞI TẠO PIPELINE DỮ LIỆU & TỪ ĐIỂN")
    print("=" * 60)

    # 1. Đọc tập train sơ bộ để tiền xử lý và build Vocabulary
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Không tìm thấy tập train tại: {train_path}")

    df_train = pd.read_csv(train_path)
    train_raw_texts = df_train["text"].fillna("").astype(str).tolist()

    print(f"[*] Tiền xử lý tập train để xây dựng Vocabulary ({len(train_raw_texts):,} câu)...")
    train_processed_for_vocab = [
        preprocess_pipeline(
            t,
            use_segment=use_word_segment,
            normalize_noise=normalize_noise
        )
        for t in train_raw_texts
    ]

    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(train_processed_for_vocab)

    # 2. Khởi tạo các Dataset
    print("\n[*] Khởi tạo Train Dataset...")
    train_dataset = VietnameseSentimentDataset(
        csv_path=train_path,
        vocab=vocab,
        max_len=max_len,
        use_word_segment=use_word_segment,
        normalize_noise=normalize_noise
    )

    print("\n[*] Khởi tạo Validation Dataset...")
    val_dataset = VietnameseSentimentDataset(
        csv_path=val_path,
        vocab=vocab,
        max_len=max_len,
        use_word_segment=use_word_segment,
        normalize_noise=normalize_noise
    )
    print("[-] Tỷ lệ OOV trên tập Validation:")
    vocab.compute_oov_rate(val_dataset.processed_texts)

    print("\n[*] Khởi tạo Test Dataset...")
    test_dataset = VietnameseSentimentDataset(
        csv_path=test_path,
        vocab=vocab,
        max_len=max_len,
        use_word_segment=use_word_segment,
        normalize_noise=normalize_noise
    )
    print("[-] Tỷ lệ OOV trên tập Test:")
    vocab.compute_oov_rate(test_dataset.processed_texts)

    # 3. Tạo DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\n[+] Hoàn tất khởi tạo DataLoaders:")
    print(f"    - Train batches: {len(train_loader)} (tổng {len(train_dataset)} mẫu)")
    print(f"    - Val batches  : {len(val_loader)} (tổng {len(val_dataset)} mẫu)")
    print(f"    - Test batches : {len(test_loader)} (tổng {len(test_dataset)} mẫu)")
    print(f"    - Vocab size   : {len(vocab):,} tokens")
    print("=" * 60)

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    print("=== TEST MODULE sentiment_dataset.py ===")
    train_csv = os.path.join(_project_root, "data", "raw", "train.csv")
    val_csv = os.path.join(_project_root, "data", "raw", "validation.csv")
    test_csv = os.path.join(_project_root, "data", "raw", "test.csv")

    if not os.path.exists(train_csv):
        print(f"[!] File {train_csv} chưa tồn tại. Hãy chạy data/download_data.py trước.")
        sys.exit(0)

    # Test create_dataloaders
    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_path=train_csv,
        val_path=val_csv,
        test_path=test_csv,
        batch_size=32,
        max_len=64,
        min_freq=2,
        use_word_segment=True,
        normalize_noise=False
    )

    print(f"\nKích thước từ điển (Vocab size): {len(vocab)}")

    # Lấy 1 batch từ train_loader và in shape
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        print("\n--- KIỂM TRA 1 BATCH TỪ TRAIN LOADER ---")
        print(f"input_ids shape     : {input_ids.shape} (dtype={input_ids.dtype})")
        print(f"attention_mask shape: {attention_mask.shape} (dtype={attention_mask.dtype})")
        print(f"label shape         : {labels.shape} (dtype={labels.dtype})")
        print(f"Phân bố nhãn trong batch: {torch.bincount(labels).tolist()}")

        print("\nVí dụ mẫu đầu tiên trong batch:")
        first_input_ids = input_ids[0].tolist()
        print(f"Token IDs (15 token đầu): {first_input_ids[:15]}")
        print(f"Decoded text            : {vocab.decode(first_input_ids[:15])}")
        print(f"Attention Mask (15 đầu) : {attention_mask[0][:15].tolist()}")
        print(f"Label                   : {labels[0].item()}")
        break
