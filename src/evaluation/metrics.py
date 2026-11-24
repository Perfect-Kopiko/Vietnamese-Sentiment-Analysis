"""Chỉ số đánh giá theo thứ tự nhãn 0, 1, 2."""

import sys
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

LABEL_IDS = [0, 1, 2]
LABEL_NAMES = ("negative", "neutral", "positive")


def _accuracy_and_macro_f1(y_true, y_pred, *, labels=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def _confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist()


def classification_metrics(y_true, y_pred):
    """Tính trung bình macro trên cả ba nhãn, kể cả nhãn vắng mặt."""
    kwargs = {"labels": LABEL_IDS, "average": "macro", "zero_division": 0}
    return {
        **_accuracy_and_macro_f1(y_true, y_pred, labels=LABEL_IDS),
        "macro_precision": float(precision_score(y_true, y_pred, **kwargs)),
        "macro_recall": float(recall_score(y_true, y_pred, **kwargs)),
    }


def classification_details(y_true, y_pred):
    return {
        "classification_report": classification_report(
            y_true, y_pred, labels=LABEL_IDS,
            target_names=LABEL_NAMES,
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": _confusion_matrix(y_true, y_pred),
        "label_order": LABEL_IDS,
    }


def _numpy_labels(values):
    # Tensor chỉ xuất hiện khi PyTorch đã được nạp.
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.int64)


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Nhận list, mảng NumPy hoặc tensor và tính Macro-F1 trên các nhãn xuất hiện."""
    y_true = _numpy_labels(y_true)
    y_pred = _numpy_labels(y_pred)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=LABEL_IDS, zero_division=0)
    return {
        **_accuracy_and_macro_f1(y_true, y_pred),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        **{f"f1_{name}": float(score) for name, score in zip(LABEL_NAMES, f1_per_class)},
        "confusion_matrix": _confusion_matrix(y_true, y_pred),
    }


def format_metrics_report(metrics: Dict[str, Any], title: str = "Evaluation Results") -> str:
    """
    Tạo chuỗi định dạng bảng báo cáo kết quả đánh giá rõ ràng hiển thị ra terminal hoặc lưu log.

    Args:
        metrics: Dictionary kết quả từ compute_metrics.
        title: Tiêu đề báo cáo.

    Returns:
        str: Chuỗi văn bản bảng kết quả.
    """
    cm = metrics.get("confusion_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    cm_str = (
        f"       Pred Neg (0)  Pred Neu (1)  Pred Pos (2)\n"
        f"  Neg:    {cm[0][0]:<12}  {cm[0][1]:<12}  {cm[0][2]:<12}\n"
        f"  Neu:    {cm[1][0]:<12}  {cm[1][1]:<12}  {cm[1][2]:<12}\n"
        f"  Pos:    {cm[2][0]:<12}  {cm[2][1]:<12}  {cm[2][2]:<12}"
    )

    report = (
        f"\n{'=' * 65}\n"
        f"  {title.upper()}\n"
        f"{'-' * 65}\n"
        f"  Accuracy       : {metrics.get('accuracy', 0.0) * 100:.2f}%\n"
        f"  Macro F1-Score : {metrics.get('macro_f1', 0.0) * 100:.2f}%\n"
        f"  Weighted F1    : {metrics.get('weighted_f1', 0.0) * 100:.2f}%\n"
        f"  -------------------------------------------------------------\n"
        f"  F1 - Negative (0): {metrics.get('f1_negative', 0.0) * 100:.2f}%\n"
        f"  F1 - Neutral  (1): {metrics.get('f1_neutral', 0.0) * 100:.2f}%\n"
        f"  F1 - Positive (2): {metrics.get('f1_positive', 0.0) * 100:.2f}%\n"
        f"  -------------------------------------------------------------\n"
        f"  Confusion Matrix (Rows: True, Cols: Pred):\n"
        f"{cm_str}\n"
        f"{'=' * 65}\n"
    )
    return report


if __name__ == "__main__":
    print("=== TEST MODULE metrics.py ===")
    y_true_sim = [0, 1, 2, 0, 1, 2, 2, 2, 0, 1]
    y_pred_sim = [0, 2, 2, 0, 1, 2, 1, 2, 0, 1]

    results = compute_metrics(y_true_sim, y_pred_sim)
    print(format_metrics_report(results, title="Kiểm thử hàm đánh giá chuẩn"))
