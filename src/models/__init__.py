"""
Package chứa các mô hình Deep Learning và Baseline cho bài toán Phân tích cảm xúc tiếng Việt:
- BiLSTMSentiment: Kiến trúc BiLSTM tùy biến với Masked Self-Attention / Mean-Max Pooling.
- SelfAttentionPooling: Module cơ chế chú ý có mặt nạ.
- FocalLoss, compute_class_weights, get_loss_function: Các hàm mất mát xử lý mất cân bằng lớp.
"""

from importlib import import_module

# Chỉ nạp PyTorch khi truy cập API BiLSTM.
_EXPORTS = {
    "BiLSTMSentiment": ".bilstm",
    "SelfAttentionPooling": ".bilstm",
    "FocalLoss": ".losses",
    "compute_class_weights": ".losses",
    "get_loss_function": ".losses",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
