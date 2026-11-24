import numpy as np
import pytest

from src.evaluation import compute_metrics, format_metrics_report
from src.evaluation.metrics import classification_details, classification_metrics


def test_metrics_when_neutral_is_absent():
    y_true = [0, 0, 2, 2]
    y_pred = np.array([0, 2, 2, 2])

    tfidf = classification_metrics(y_true, y_pred)
    details = classification_details(y_true, y_pred)
    bilstm = compute_metrics(y_true, y_pred)

    assert tfidf["accuracy"] == bilstm["accuracy"] == 0.75
    assert tfidf["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 3)
    assert tfidf["macro_precision"] == pytest.approx((1 + 2 / 3) / 3)
    assert tfidf["macro_recall"] == pytest.approx((0.5 + 1) / 3)
    assert bilstm["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert bilstm["weighted_f1"] == pytest.approx(bilstm["macro_f1"])
    assert bilstm["f1_negative"] == pytest.approx(2 / 3)
    assert bilstm["f1_neutral"] == 0.0
    assert bilstm["f1_positive"] == pytest.approx(0.8)
    assert details["label_order"] == [0, 1, 2]
    assert details["classification_report"]["neutral"]["support"] == 0
    assert details["confusion_matrix"] == bilstm["confusion_matrix"] == [
        [1, 0, 1],
        [0, 0, 0],
        [0, 0, 2],
    ]
    assert "73.33%" in format_metrics_report(bilstm)


def test_torch_tensor_metrics_and_bilstm_exports():
    torch = pytest.importorskip("torch")

    from src.datasets import Vocabulary, create_dataloaders
    from src.datasets.sentiment_dataset import create_dataloaders as dataset_loader
    from src.datasets.vocab import Vocabulary as DatasetVocabulary
    from src.models import BiLSTMSentiment, FocalLoss
    from src.models.bilstm import BiLSTMSentiment as BiLSTMModel
    from src.models.losses import FocalLoss as BiLSTMLoss

    y_true = [0, 1, 2]
    y_pred = [0, 2, 2]

    assert compute_metrics(torch.tensor(y_true), torch.tensor(y_pred)) == compute_metrics(
        y_true, y_pred
    )
    assert Vocabulary is DatasetVocabulary
    assert create_dataloaders is dataset_loader
    assert BiLSTMSentiment is BiLSTMModel
    assert FocalLoss is BiLSTMLoss
