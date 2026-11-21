"""
Module tính toán và định dạng các chỉ số đánh giá (Evaluation Metrics)
cho bài toán Phân tích cảm xúc văn bản tiếng Việt (3 lớp: Negative=0, Neutral=1, Positive=2).
"""

from typing import Any, Dict, List, Union

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def compute_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor]
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số đánh giá chuẩn cho bài toán phân loại cảm xúc 3 lớp:
    - accuracy: Độ chính xác tổng thể.
    - macro_f1: F1-score trung bình không trọng số giữa các lớp (quan trọng nhất khi mất cân bằng).
    - weighted_f1: F1-score có trọng số theo kích thước từng lớp.
    - f1_negative (lớp 0), f1_neutral (lớp 1), f1_positive (lớp 2): F1 riêng từng lớp.
    - confusion_matrix: Ma trận nhầm lẫn kích thước 3x3 dạng danh sách lồng.

    Args:
        y_true: Nhãn thực tế (ground truth).
        y_pred: Nhãn dự đoán (predictions).

    Returns:
        dict: Chứa đầy đủ các chỉ số đánh giá.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # F1 score cho từng lớp (0: Negative, 1: Neutral, 2: Positive)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    f1_neg = float(f1_per_class[0])
    f1_neu = float(f1_per_class[1])
    f1_pos = float(f1_per_class[2])

    # Confusion matrix (3x3)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "f1_negative": f1_neg,
        "f1_neutral": f1_neu,
        "f1_positive": f1_pos,
        "confusion_matrix": cm
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
