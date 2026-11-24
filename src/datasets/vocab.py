"""
Xây dựng lớp từ điển (Vocabulary) và hàm nạp pre-trained Word2Vec embeddings cho tiếng Việt.
Hỗ trợ nạp streaming theo từng dòng từ file Word2Vec (.vec / .txt) giúp tiết kiệm RAM.
"""

import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class Vocabulary:
    """
    Quản lý từ điển ánh xạ giữa từ (token) và chỉ số (index).
    Cố định 2 token đặc biệt:
    - <PAD>: index 0 (token đệm chuỗi)
    - <UNK>: index 1 (token đại diện từ chưa biết / ngoài từ điển)
    """

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq: int = 2):
        """
        Khởi tạo Vocabulary.

        Args:
            min_freq: Tần suất xuất hiện tối thiểu trong tập văn bản để được đưa vào từ điển.
        """
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        self.idx2word: Dict[int, str] = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN
        }
        self.word_counts: Counter = Counter()

    def __len__(self) -> int:
        return len(self.word2idx)

    def build_vocab(self, texts: List[str]) -> None:
        """
        Xây dựng từ điển từ danh sách các câu đã được tokenize/tiền xử lý.

        Args:
            texts: Danh sách chuỗi văn bản (các từ phân tách bằng khoảng trắng).
        """
        self.word_counts = Counter()
        for text in texts:
            if not text:
                continue
            words = text.strip().split()
            self.word_counts.update(words)

        # Reset lại từ điển giữ 2 token đặc biệt
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX
        }
        self.idx2word = {
            self.PAD_IDX: self.PAD_TOKEN,
            self.UNK_IDX: self.UNK_TOKEN
        }

        # Thêm các từ đạt ngưỡng tần suất min_freq
        # Sắp xếp theo tần suất giảm dần để chỉ số ổn định
        sorted_words = sorted(
            [word for word, count in self.word_counts.items() if count >= self.min_freq],
            key=lambda w: (-self.word_counts[w], w)
        )

        for word in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(
            f"[*] Đã xây dựng Vocab: tổng {len(self.word_counts):,} từ xuất hiện, "
            f"{len(self.word2idx):,} từ được lưu vào từ điển (min_freq={self.min_freq})."
        )

    def encode(self, text: str) -> List[int]:
        """
        Chuyển chuỗi văn bản thành danh sách token ID.
        Các từ không có trong từ điển được ánh xạ sang UNK_IDX.

        Args:
            text: Chuỗi văn bản phân tách bằng khoảng trắng.

        Returns:
            Danh sách chỉ số nguyên tương ứng.
        """
        if not text:
            return []
        words = text.strip().split()
        return [self.word2idx.get(w, self.UNK_IDX) for w in words]

    def decode(self, indices: List[int]) -> str:
        """
        Giải mã danh sách token ID thành chuỗi văn bản.

        Args:
            indices: Danh sách chỉ số nguyên.

        Returns:
            Chuỗi văn bản ghép lại bởi khoảng trắng.
        """
        if not indices:
            return ""
        words = [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]
        return " ".join(words)

    def compute_oov_rate(self, texts: List[str], verbose: bool = True) -> float:
        """
        Thống kê tỷ lệ từ ngoài từ điển (OOV rate) trên một tập văn bản (validation/test).

        Args:
            texts: Danh sách văn bản kiểm tra.
            verbose: Có in kết quả ra màn hình hay không.

        Returns:
            Tỷ lệ OOV dạng float (0.0 đến 1.0).
        """
        total_tokens = 0
        oov_tokens = 0

        for text in texts:
            if not text:
                continue
            words = text.strip().split()
            for w in words:
                total_tokens += 1
                if w not in self.word2idx:
                    oov_tokens += 1

        oov_rate = (oov_tokens / total_tokens) if total_tokens > 0 else 0.0
        if verbose:
            print(
                f"[*] Thống kê OOV: {oov_tokens:,} / {total_tokens:,} tokens "
                f"({oov_rate * 100:.2f}%) nằm ngoài từ điển."
            )
        return oov_rate

    def save(self, path: str) -> None:
        """
        Lưu thông tin Vocabulary vào file JSON.

        Args:
            path: Đường dẫn tệp lưu (.json).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "min_freq": self.min_freq,
            "word2idx": self.word2idx,
            "word_counts": dict(self.word_counts)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[+] Đã lưu Vocabulary ({len(self.word2idx)} tokens) vào: {path}")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """
        Khôi phục đối tượng Vocabulary từ file JSON.

        Args:
            path: Đường dẫn tệp json đã lưu.

        Returns:
            Đối tượng Vocabulary đã nạp lại.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file Vocabulary tại: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = cls(min_freq=data.get("min_freq", 2))
        vocab.word2idx = data["word2idx"]
        # Phục hồi idx2word đảm bảo key dạng int
        vocab.idx2word = {int(idx): word for word, idx in vocab.word2idx.items()}
        vocab.word_counts = Counter(data.get("word_counts", {}))
        print(f"[+] Đã nạp Vocabulary ({len(vocab)} tokens) từ: {path}")
        return vocab


def load_word2vec_embeddings(
    vocab: Vocabulary,
    w2v_path: Optional[str] = None,
    embed_dim: int = 300
) -> np.ndarray:
    """
    Nạp ma trận embedding từ file Word2Vec định dạng text (.vec hoặc .txt, ví dụ PhoW2V).
    
    Quy trình tối ưu bộ nhớ (Streaming line-by-line):
    1. Khởi tạo ma trận ngẫu nhiên N(0, 0.1) kích thước (len(vocab), embed_dim).
    2. Riêng vector tại hàng PAD_IDX (0) gán bằng toàn số 0.0.
    3. Đọc từng dòng của file pre-trained: chỉ parse vector khi token nằm trong vocab.word2idx,
       giúp không bị tràn bộ nhớ RAM (đặc biệt khi RAM nhỏ <= 8GB).
    4. Các từ OOV trong vocab giữ nguyên vector ngẫu nhiên ban đầu.

    Args:
        vocab: Đối tượng Vocabulary đã được build trên tập dữ liệu.
        w2v_path: Đường dẫn tới file Word2Vec định dạng text (ví dụ 'word2vec.vec').
        embed_dim: Số chiều embedding (mặc định 300).

    Returns:
        numpy.ndarray: Ma trận embedding shape (len(vocab), embed_dim) kiểu float32.
    """
    vocab_size = len(vocab)
    # Khởi tạo ma trận ngẫu nhiên theo phân phối chuẩn N(0, 0.1)
    embedding_matrix = np.random.normal(0.0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
    # Hàng 0 (<PAD>) luôn là vector 0.0
    embedding_matrix[Vocabulary.PAD_IDX] = np.zeros(embed_dim, dtype=np.float32)

    if not w2v_path or not os.path.exists(w2v_path):
        print(
            f"[!] Đường dẫn Word2Vec '{w2v_path}' không tồn tại hoặc rỗng. "
            f"Trả về ma trận khởi tạo ngẫu nhiên kích thước: {embedding_matrix.shape}"
        )
        return embedding_matrix

    print(f"[*] Đang nạp Word2Vec (streaming line-by-line) từ '{w2v_path}'...")
    loaded_words = 0

    try:
        with open(w2v_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip().split()
            # Kiểm tra xem dòng đầu tiên có phải là header (num_words dim) hay không
            if len(first_line) == 2:
                try:
                    total_w2v_words, file_dim = int(first_line[0]), int(first_line[1])
                    print(f"[*] File header: {total_w2v_words:,} từ, chiều vector: {file_dim}")
                    if file_dim != embed_dim:
                        print(
                            f"[!] Cảnh báo: Chiều vector trong file ({file_dim}) "
                            f"khác với embed_dim yêu cầu ({embed_dim}). Cập nhật embed_dim={file_dim}."
                        )
                        embed_dim = file_dim
                        embedding_matrix = np.random.normal(0.0, 0.1, size=(vocab_size, embed_dim)).astype(np.float32)
                        embedding_matrix[Vocabulary.PAD_IDX] = np.zeros(embed_dim, dtype=np.float32)
                except ValueError:
                    # Không phải header, mà là 1 dòng chứa từ có độ dài 2
                    word = first_line[0]
                    if word in vocab.word2idx:
                        vec = np.asarray(first_line[1:], dtype=np.float32)
                        if len(vec) == embed_dim:
                            embedding_matrix[vocab.word2idx[word]] = vec
                            loaded_words += 1
            elif len(first_line) > 2:
                word = first_line[0]
                if word in vocab.word2idx:
                    vec = np.asarray(first_line[1:], dtype=np.float32)
                    if len(vec) == embed_dim:
                        embedding_matrix[vocab.word2idx[word]] = vec
                        loaded_words += 1

            # Đọc từng dòng tiếp theo
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                word = parts[0]
                if word in vocab.word2idx:
                    try:
                        vector = np.asarray(parts[1:], dtype=np.float32)
                        if len(vector) == embed_dim:
                            idx = vocab.word2idx[word]
                            embedding_matrix[idx] = vector
                            loaded_words += 1
                    except ValueError:
                        continue

    except Exception as exc:
        print(f"[!] Lỗi khi đọc file Word2Vec: {exc}. Giữ nguyên ma trận khởi tạo.")

    # Đảm bảo hàng PAD luôn là 0.0
    embedding_matrix[Vocabulary.PAD_IDX] = np.zeros(embed_dim, dtype=np.float32)

    match_rate = (loaded_words / vocab_size) * 100 if vocab_size > 0 else 0.0
    print(
        f"[+] Đã khớp thành công: {loaded_words:,} / {vocab_size:,} từ trong Vocab "
        f"({match_rate:.2f}%). Các từ còn lại được giữ khởi tạo ngẫu nhiên."
    )

    return embedding_matrix


if __name__ == "__main__":
    print("=== TEST MODULE vocab.py ===")
    sample_texts = [
        "thầy giảng_viên dạy bài_tập rất hay",
        "sinh_viên học bài_tập rất chăm_chỉ",
        "thầy rất nhiệt_tình với sinh_viên",
        "kiểm_tra môn học rất khó"
    ]

    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(sample_texts)

    test_sentence = "thầy dạy nhiệt_tình nhưng kiểm_tra rất khó"
    encoded = vocab.encode(test_sentence)
    decoded = vocab.decode(encoded)
    print(f"Câu test gốc: {test_sentence}")
    print(f"Encoded IDs : {encoded}")
    print(f"Decoded     : {decoded}")

    vocab.compute_oov_rate(["thầy dạy môn mới tinh lạ_lẫm"])

    # Test load_word2vec_embeddings với đường dẫn giả lập
    matrix = load_word2vec_embeddings(vocab, w2v_path="non_existing_w2v.vec", embed_dim=100)
    print(f"Shape ma trận embedding: {matrix.shape}")
    print(f"Hàng PAD (0) có phải toàn 0? {np.all(matrix[0] == 0.0)}")
