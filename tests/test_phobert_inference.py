"""Đối chiếu checkpoint PhoBERT thật với pipeline huấn luyện và demo offline."""

import socket
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from demo.app import EXAMPLES
from demo.inference_pipeline import InferencePipeline, MODEL_NAMES
from demo.phobert_inference import PhoBERTInference
from src.models.tfidf_baseline import LABELS, ROOT
from src.preprocessing import preprocess_pipeline
from src.trainers.phobert_trainer import SentimentDataset


@pytest.fixture(scope="module")
def offline_pipeline():
    checkpoint = ROOT / "checkpoints/best_phobert/model.pt"
    if not checkpoint.is_file():
        pytest.skip("Copy the verified PhoBERT checkpoint and tokenizer before integration tests")
    with pytest.MonkeyPatch.context() as patch:
        loader = Mock(wraps=torch.load)
        patch.setattr(torch, "load", loader)
        patch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("Network access")))
        inference = InferencePipeline()
        assert inference.phobert is not None, inference.status("PhoBERT")
        yield inference, loader


@pytest.mark.parametrize("normalize", [False, True])
def test_encoding_and_probabilities_match_training_dataset(offline_pipeline, normalize):
    predictor = offline_pipeline[0].phobert
    texts = [*EXAMPLES.values(), "<b>GV</b> ko tốt!!! https://example.com",
             "giảng viên nhiệt tình " * 80, "<br>"]
    processed = [preprocess_pipeline(text, use_segment=True, normalize_noise=normalize) for text in texts]
    dataset = SentimentDataset(processed, [0] * len(texts), predictor.tokenizer, max_length=128)
    inputs = {key: torch.stack([dataset[i][key] for i in range(len(dataset))])
              for key in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        expected = predictor.model(**inputs).softmax(-1)
    for index, text in enumerate(texts):
        encoded = predictor.encode(text, normalize)
        for key in inputs:
            torch.testing.assert_close(encoded[key][0], inputs[key][index])
        np.testing.assert_allclose(predictor.predict_proba(text, normalize), expected[index].numpy(),
                                   rtol=1e-4, atol=1e-6)
    assert predictor.max_len == 128
    assert predictor.class_ids == (0, 1, 2)
    assert predictor.model.training is False
    assert all(t.device.type == "cpu" for t in predictor.model.parameters())
    assert all(t.device.type == "cpu" for t in predictor.model.buffers())
    assert all(not t.requires_grad for t in predictor.model.parameters())


def test_default_uses_checkpoint_normalization(offline_pipeline):
    predictor = offline_pipeline[0].phobert
    text = EXAMPLES["Teencode"]
    np.testing.assert_array_equal(predictor.predict_proba(text), predictor.predict_proba(text, True))


def test_three_models_toggle_and_load_once_without_network(offline_pipeline):
    inference, loader = offline_pipeline
    identities = (id(inference.artifact), id(inference.bilstm.model), id(inference.phobert.model))
    for text in EXAMPLES.values():
        first = inference.compare(text, False)
        enabled = inference.compare(text, True)
        assert inference.compare(text, False) == first
        for results in (first, enabled):
            assert [r["model"] for r in results] == list(MODEL_NAMES)
            for result in results:
                assert result["error"] is None
                assert set(result["probabilities"]) == set(LABELS.values())
                assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
                assert LABELS[result["label_id"]] == result["label"]
                assert max(result["probabilities"], key=result["probabilities"].get) == result["label"]
        if text == EXAMPLES["Teencode"]:
            for off, on in zip(first, enabled):
                assert off["probabilities"] != on["probabilities"]
            assert [r["label_id"] for r in first] == [2, 2, 0]
            assert [r["label_id"] for r in enabled] == [0, 0, 0]
    assert identities == (id(inference.artifact), id(inference.bilstm.model), id(inference.phobert.model))
    for path in (inference.bilstm_checkpoint, inference.phobert_checkpoint):
        assert sum(call.args[0] == path for call in loader.call_args_list) == 1


def test_phobert_errors_do_not_disable_other_models(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"broken")
    inference = InferencePipeline(phobert_checkpoint=checkpoint)
    assert "Không nạp được PhoBERT" in inference.status("PhoBERT")
    assert inference.predict("Thầy giảng hay", "PhoBERT")["probabilities"] is None
    assert inference.predict("Thầy giảng hay")["probabilities"]
    assert inference.predict("Thầy giảng hay", "BiLSTM")["probabilities"]


def test_missing_tokenizer_is_reported(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    with pytest.raises(FileNotFoundError, match="config.json"):
        PhoBERTInference(checkpoint)
