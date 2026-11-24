"""
Kiến trúc mạng BiLSTM tùy biến (Custom BiLSTM with Masked Pooling / Self-Attention)
phục vụ Đề tài 6: Phân tích cảm xúc văn bản tiếng Việt.

Tác giả / Sinh viên thực hiện: Nhóm nghiên cứu Học Sâu - Đề tài 6.
Kiến trúc hỗ trợ:
1. Embedding Layer: Tùy chọn nạp pre-trained Word2Vec (PhoW2V) hoặc học từ đầu, hỗ trợ freeze.
2. Bidirectional LSTM: Xử lý ngữ cảnh 2 chiều từ trái sang phải và phải sang trái.
3. Cơ chế Pooling:
   - "attention": Self-Attention Pooling có mặt nạ (Masked Attention) giúp tập trung vào
     các từ khóa mang sắc thái cảm xúc, triệt tiêu trọng số của token đệm <PAD>.
   - "mean_max": Kết hợp Masked Mean Pooling và Masked Max Pooling (Vector nối đôi 2 * hidden_dim * 2).
4. Phân loại cảm xúc: LayerNorm + Dropout + Linear ra 3 lớp cảm xúc (Negative, Neutral, Positive).
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionPooling(nn.Module):
    """
    Cơ chế Self-Attention Pooling có mặt nạ (Masked Self-Attention) cho chuỗi đầu ra BiLSTM.

    Cách hoạt động:
    1. Chiếu các hidden states h_t qua tầng tuyến tính: u_t = tanh(W * h_t + b).
    2. Chiếu u_t với vector ngữ cảnh v_a để tạo điểm chú ý vô hướng: e_t = v_a^T * u_t.
    3. Áp dụng attention_mask: Gán e_t = -1e9 tại các vị trí <PAD> (mask == 0)
       để xác suất softmax tại các vị trí này triệt tiêu hoàn toàn về 0.
    4. Trọng số chú ý: a_t = softmax(e_t).
    5. Vector ngữ cảnh: c = sum(a_t * h_t).
    """

    def __init__(self, hidden_dim: int):
        """
        Khởi tạo SelfAttentionPooling.

        Args:
            hidden_dim: Kích thước biểu diễn của BiLSTM (thường là hidden_dim * 2 nếu 2 chiều).
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(
        self,
        lstm_outputs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính toán vector ngữ cảnh và ma trận trọng số chú ý.

        Args:
            lstm_outputs: Tensor đầu ra từ BiLSTM, shape [batch_size, seq_len, hidden_dim].
            attention_mask: Tensor mặt nạ nhị phân, shape [batch_size, seq_len] (1: token thật, 0: <PAD>).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - context_vector: shape [batch_size, hidden_dim]
                - attn_weights: shape [batch_size, seq_len]
        """
        # lstm_outputs: [B, L, H]
        # energy: [B, L, 1] -> squeeze về [B, L]
        energy = self.projection(lstm_outputs).squeeze(-1)

        # Áp dụng attention mask: Gán giá trị rất nhỏ (-1e9) tại vị trí mask == 0 (<PAD>)
        # để softmax tại đó triệt tiêu về 0
        if attention_mask is not None:
            mask = attention_mask.bool()
            # Điểm tại mask == False được gán -1e9
            energy = energy.masked_fill(~mask, -1e9)

        # Tính trọng số chú ý qua Softmax theo chiều chiều dài câu
        attn_weights = F.softmax(energy, dim=-1)  # [B, L]

        # Đề phòng trường hợp cả câu toàn PAD (tất cả bị gán -1e9 sinh ra nan), fill lại 0
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Nhân trọng số với biểu diễn LSTM: [B, 1, L] @ [B, L, H] -> [B, 1, H] -> [B, H]
        context_vector = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)

        return context_vector, attn_weights


class BiLSTMSentiment(nn.Module):
    """
    Kiến trúc BiLSTM phân loại cảm xúc văn bản tiếng Việt.
    Hỗ trợ 2 cơ chế trích chọn đặc trưng chuỗi (pooling):
    1. 'attention': Self-Attention Pooling có mặt nạ.
    2. 'mean_max': Kết hợp Masked Mean Pooling và Masked Max Pooling.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.3,
        pooling: str = "attention",
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False
    ):
        """
        Khởi tạo BiLSTMSentiment.

        Args:
            vocab_size: Kích thước từ điển.
            embed_dim: Số chiều vector nhúng từ.
            hidden_dim: Kích thước trạng thái ẩn của LSTM mỗi chiều.
            num_layers: Số tầng LSTM xếp chồng (stacked LSTM).
            num_classes: Số lượng nhãn phân loại (mặc định 3).
            dropout: Tỷ lệ dropout áp dụng giữa các tầng và trước classifier.
            pooling: Cơ chế pooling ('attention' hoặc 'mean_max').
            pretrained_embeddings: Ma trận numpy trọng số Word2Vec tiền huấn luyện.
            freeze_embeddings: Nếu True thì đóng băng không cập nhật trọng số embedding.
        """
        super().__init__()
        self.pooling = pooling.lower()
        if self.pooling not in ["attention", "mean_max"]:
            raise ValueError(f"Cơ chế pooling '{pooling}' không hợp lệ. Chọn 'attention' hoặc 'mean_max'.")

        # 1. Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            tensor_embeddings = torch.tensor(pretrained_embeddings, dtype=torch.float32)
            self.embedding.weight.data.copy_(tensor_embeddings)
            print(f"[*] Đã nạp pre-trained embeddings vào BiLSTM shape: {pretrained_embeddings.shape}")
        
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            print("[*] Đã đóng băng (freeze) trọng số Embedding layer.")

        self.embed_dropout = nn.Dropout(dropout)

        # 2. Bidirectional LSTM Layer
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=lstm_dropout
        )

        bilstm_out_dim = hidden_dim * 2

        # 3. Pooling Mechanism
        if self.pooling == "attention":
            self.attention_pool = SelfAttentionPooling(hidden_dim=bilstm_out_dim)
            classifier_in_dim = bilstm_out_dim
        else:  # mean_max
            self.attention_pool = None
            classifier_in_dim = bilstm_out_dim * 2  # Nối mean pooling (H) và max pooling (H)

        # 4. Classifier Head
        self.layer_norm = nn.LayerNorm(classifier_in_dim)
        self.classifier_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(classifier_in_dim, num_classes)

    def _masked_mean_max_pooling(
        self,
        lstm_outputs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Masked Mean Pooling kết hợp Masked Max Pooling:
        - Mean Pooling: Chỉ tính trung bình trên các vị trí mask == 1.
        - Max Pooling: Gán -1e9 tại các vị trí mask == 0 trước khi lấy max.
        """
        # lstm_outputs: [B, L, H]
        # attention_mask: [B, L]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]

        # 1. Masked Mean Pooling
        sum_embeddings = (lstm_outputs * mask).sum(dim=1)  # [B, H]
        lengths = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        mean_pooled = sum_embeddings / lengths  # [B, H]

        # 2. Masked Max Pooling
        # Gán giá trị cực nhỏ tại các vị trí <PAD> để không bị chọn làm max
        masked_outputs = lstm_outputs.masked_fill(attention_mask.unsqueeze(-1) == 0, -1e9)
        max_pooled, _ = masked_outputs.max(dim=1)  # [B, H]

        # Nối 2 vector đặc trưng
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # [B, H * 2]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Thực hiện lan truyền tiến (Forward Pass).

        Args:
            input_ids: Tensor chứa ID từ vựng, shape [batch_size, seq_len].
            attention_mask: Tensor mặt nạ [batch_size, seq_len] (1 cho token thật, 0 cho <PAD>).
            return_attention: Có trả về ma trận trọng số attention hay không (chỉ dùng với pooling="attention").

        Returns:
            logits: Tensor dự đoán phân loại, shape [batch_size, num_classes].
            (Tùy chọn) attn_weights: Tensor trọng số chú ý, shape [batch_size, seq_len].
        """
        # [B, L] -> [B, L, embed_dim]
        embedded = self.embedding(input_ids)
        embedded = self.embed_dropout(embedded)

        # Qua BiLSTM: outputs shape [B, L, hidden_dim * 2]
        lstm_outputs, _ = self.lstm(embedded)

        attn_weights = None
        if self.pooling == "attention":
            features, attn_weights = self.attention_pool(lstm_outputs, attention_mask)
        else:
            features = self._masked_mean_max_pooling(lstm_outputs, attention_mask)

        # Classifier head
        normed_features = self.layer_norm(features)
        dropped_features = self.classifier_dropout(normed_features)
        logits = self.fc(dropped_features)  # [B, num_classes]

        if return_attention:
            return logits, attn_weights
        return logits


if __name__ == "__main__":
    print("=== TEST MODULE bilstm.py ===")
    import sys
    import os

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from src.models.losses import get_loss_function, compute_class_weights

    batch_size = 4
    seq_len = 64
    vocab_size = 1000
    embed_dim = 300
    hidden_dim = 128
    num_classes = 3

    # Giả lập input_ids ngẫu nhiên từ 1 đến vocab_size - 1
    dummy_input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), dtype=torch.long)
    # Giả lập các độ dài câu khác nhau: 10, 25, 40, 64 token thật
    lengths = [10, 25, 40, 64]
    dummy_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    for i, length in enumerate(lengths):
        dummy_mask[i, :length] = 1
        dummy_input_ids[i, length:] = 0  # Gán token <PAD>

    dummy_targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    # -------------------------------------------------------------
    # 1. Test Model với pooling="attention"
    # -------------------------------------------------------------
    print("\n--- 1. Kiểm tra Model với pooling='attention' ---")
    model_attn = BiLSTMSentiment(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        pooling="attention"
    )

    logits_attn, attn_w = model_attn(dummy_input_ids, dummy_mask, return_attention=True)
    print(f"Logits shape       : {logits_attn.shape} (kỳ vọng: [{batch_size}, {num_classes}])")
    print(f"Attn weights shape : {attn_w.shape} (kỳ vọng: [{batch_size}, {seq_len}])")

    # Kiểm tra tổng trọng số attention trên mỗi câu = 1.0 và vị trí <PAD> = 0.0
    attn_sums = attn_w.sum(dim=-1).detach().numpy()
    print(f"Tổng trọng số attention mỗi câu: {np.round(attn_sums, 4)} (kỳ vọng: toàn 1.0)")
    assert np.allclose(attn_sums, 1.0, atol=1e-5), "Lỗi: Tổng attention weights khác 1.0!"

    for i, length in enumerate(lengths):
        pad_weights = attn_w[i, length:].detach().numpy()
        if len(pad_weights) > 0:
            max_pad_val = np.max(pad_weights)
            print(f"  Câu {i} (độ dài {length}/{seq_len}): Max attention tại vị trí <PAD> = {max_pad_val:.2e}")
            assert np.allclose(pad_weights, 0.0, atol=1e-6), f"Lỗi: Token <PAD> ở câu {i} có trọng số khác 0!"

    # -------------------------------------------------------------
    # 2. Test Model với pooling="mean_max"
    # -------------------------------------------------------------
    print("\n--- 2. Kiểm tra Model với pooling='mean_max' ---")
    model_mean_max = BiLSTMSentiment(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
        pooling="mean_max"
    )

    logits_mm = model_mean_max(dummy_input_ids, dummy_mask)
    print(f"Logits shape       : {logits_mm.shape} (kỳ vọng: [{batch_size}, {num_classes}])")

    # -------------------------------------------------------------
    # 3. Test Hàm mất mát (Loss functions) với logits vừa tạo
    # -------------------------------------------------------------
    print("\n--- 3. Kiểm tra tính Loss với cả 3 hàm mất mát ---")
    class_weights = compute_class_weights([0, 1, 1, 2, 2, 2, 2, 2], num_classes=3)

    loss_ce = get_loss_function("ce")(logits_attn, dummy_targets)
    loss_wce = get_loss_function("weighted_ce", class_weights=class_weights)(logits_attn, dummy_targets)
    loss_focal = get_loss_function("focal", class_weights=class_weights, focal_gamma=2.0)(logits_attn, dummy_targets)

    print(f"Loss CE         : {loss_ce.item():.4f}")
    print(f"Loss Weighted CE: {loss_wce.item():.4f}")
    print(f"Loss Focal      : {loss_focal.item():.4f}")
    print("\n[+] TẤT CẢ KIỂM THỬ ARCHITECTURE & LOSS ĐỀU VƯỢT QUA XUẤT SẮC!")
