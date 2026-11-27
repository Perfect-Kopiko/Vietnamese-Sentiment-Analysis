"""
Package quản lý Dataset và Vocabulary cho bài toán Phân tích cảm xúc tiếng Việt.
"""

from importlib import import_module

# Cho phép đọc dữ liệu TF-IDF mà không cần nạp PyTorch.
_EXPORTS = {
    "Vocabulary": ".vocab",
    "load_word2vec_embeddings": ".vocab",
    "VietnameseSentimentDataset": ".sentiment_dataset",
    "create_dataloaders": ".sentiment_dataset",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
