"""Đối chiếu suy luận với Dataset và checkpoint BiLSTM thật."""

import pandas as pd
import pytest
from unittest.mock import Mock

torch = pytest.importorskip("torch")

from demo.bilstm_inference import BiLSTMInference
from demo.inference_pipeline import InferencePipeline
from src.datasets.sentiment_dataset import VietnameseSentimentDataset
from src.models.tfidf_baseline import ROOT


@pytest.fixture(scope="module", params=[
    "bilstm_no_seg_w2v_best.pt", "bilstm_seg_w2v_weighted_ce_best.pt",
])
def predictor(request):
    candidates = [ROOT / "checkpoints/bilstm" / request.param, ROOT / request.param]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        pytest.skip(f"Missing local checkpoint: {request.param}")
    return BiLSTMInference(path)


@pytest.mark.parametrize("normalize", [False, True])
def test_predictions_match_training_dataset(predictor, tmp_path, normalize):
    texts = ["<b>GV</b> ko rõ ràng!!! https://example.com", "môn này có ba tín chỉ",
             "giảng viên nhiệt tình " * 80, "từ_lạ_98765", "<br>"]
    path = tmp_path / "samples.csv"
    pd.DataFrame({"text": texts, "label": [0, 1, 2, 1, 1]}).to_csv(path, index=False)
    dataset = VietnameseSentimentDataset(
        str(path), predictor.vocab, max_len=predictor.max_len,
        use_word_segment=predictor.data_config["use_word_segment"], normalize_noise=normalize,
    )
    batch = [dataset[i] for i in range(len(texts))]
    input_ids = torch.stack([sample["input_ids"] for sample in batch])
    masks = torch.stack([sample["attention_mask"] for sample in batch])
    with torch.inference_mode():
        expected = predictor.model(input_ids, masks).softmax(-1)
    for i, text in enumerate(texts):
        actual_ids, actual_mask = predictor.encode(text, normalize)
        torch.testing.assert_close(actual_ids[0], input_ids[i])
        torch.testing.assert_close(actual_mask[0], masks[i])
        torch.testing.assert_close(torch.from_numpy(predictor.predict_proba(text, normalize)),
                                   expected[i], rtol=1e-5, atol=1e-6)
    assert predictor.max_len == 64
    assert predictor.model.training is False
    assert next(predictor.model.parameters()).device.type == "cpu"


def test_main_checkpoint_and_repeatable_toggle(monkeypatch):
    loader = Mock(wraps=torch.load)
    monkeypatch.setattr(torch, "load", loader)
    inference = InferencePipeline()
    if not inference.bilstm_checkpoint.is_file():
        pytest.skip(inference.status("BiLSTM"))
    assert inference.bilstm is not None, inference.status("BiLSTM")
    assert inference.bilstm_checkpoint.name == "bilstm_no_seg_w2v_best.pt"
    assert inference.bilstm.data_config["use_word_segment"] is False
    text = "gv dạy ko rõ ràng, sv k hiểu bài"
    first = inference.predict(text, "BiLSTM")
    enabled = inference.predict(text, "BiLSTM", True)
    assert sum(first["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert first["probabilities"] != enabled["probabilities"]
    assert inference.predict(text, "BiLSTM") == first
    loader.assert_called_once_with(inference.bilstm_checkpoint, map_location="cpu", weights_only=True)
