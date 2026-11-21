"""
Package chứa các mô hình Deep Learning và Baseline cho bài toán Phân tích cảm xúc tiếng Việt:
- BiLSTMSentiment: Kiến trúc BiLSTM tùy biến với Masked Self-Attention / Mean-Max Pooling.
- SelfAttentionPooling: Module cơ chế chú ý có mặt nạ.
- FocalLoss, compute_class_weights, get_loss_function: Các hàm mất mát xử lý mất cân bằng lớp.
"""

from src.models.losses import (
    FocalLoss,
    compute_class_weights,
    get_loss_function,
)
from src.models.bilstm import (
    BiLSTMSentiment,
    SelfAttentionPooling,
)

__all__ = [
    "BiLSTMSentiment",
    "SelfAttentionPooling",
    "FocalLoss",
    "compute_class_weights",
    "get_loss_function",
]
