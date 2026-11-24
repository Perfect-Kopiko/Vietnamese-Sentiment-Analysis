"""Chỉ số đánh giá theo thứ tự nhãn 0, 1, 2."""

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score

LABEL_IDS = [0, 1, 2]


def classification_metrics(y_true, y_pred):
    kwargs = {"labels": LABEL_IDS, "average": "macro", "zero_division": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, **kwargs)),
        "macro_precision": float(precision_score(y_true, y_pred, **kwargs)),
        "macro_recall": float(recall_score(y_true, y_pred, **kwargs)),
    }


def classification_details(y_true, y_pred):
    return {
        "classification_report": classification_report(
            y_true, y_pred, labels=LABEL_IDS,
            target_names=["negative", "neutral", "positive"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_IDS).tolist(),
        "label_order": LABEL_IDS,
    }
