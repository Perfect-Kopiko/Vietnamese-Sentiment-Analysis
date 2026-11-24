"""Kiểm tra giao diện và dự đoán của demo."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from demo import inference_pipeline as inference_module
from demo.inference_pipeline import InferencePipeline, MAX_TEXT_LENGTH
from src.models.tfidf_baseline import LABELS, ROOT, load_checkpoint


@pytest.fixture(scope="module")
def real_artifact():
    path = ROOT / "checkpoints/best_tfidf.pkl"
    if not path.is_file():
        pytest.skip("Copy the verified baseline checkpoint before running integration tests")
    return load_checkpoint(path)


def test_mapping_uses_checkpoint_class_order(real_artifact, tmp_path):
    artifact = deepcopy(real_artifact)
    pipeline = artifact["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    permutation = [2, 0, 1]
    classifier.classes_ = classifier.classes_[permutation]
    classifier.coef_ = classifier.coef_[permutation]
    classifier.intercept_ = classifier.intercept_[permutation]
    inference = InferencePipeline(tmp_path / "absent.pkl")
    inference.artifact = artifact

    text = "Thầy dạy rất nhiệt tình và dễ hiểu."
    result = inference.predict(text)
    probabilities = pipeline.predict_proba([text])[0]
    for label, probability in zip(classifier.classes_, probabilities):
        assert result["probabilities"][LABELS[int(label)]] == pytest.approx(probability)
    expected_label = int(classifier.classes_[probabilities.argmax()])
    assert result["label_id"] == expected_label
    assert result["label"] == LABELS[expected_label]
    assert sum(result["probabilities"].values()) == pytest.approx(1)


def test_model_loaded_once(real_artifact, monkeypatch):
    loader = Mock(wraps=inference_module.load_checkpoint)
    monkeypatch.setattr(inference_module, "load_checkpoint", loader)
    inference = InferencePipeline()
    assert inference.predict("Thầy dạy hay")["probabilities"]
    assert inference.predict("Thầy dạy khó hiểu")["probabilities"]
    loader.assert_called_once_with(inference.checkpoint)


def test_saved_preprocessor_receives_raw_text_and_errors_are_reported(real_artifact, monkeypatch):
    inference = InferencePipeline()
    transform = Mock(side_effect=RuntimeError("inference error"))
    monkeypatch.setattr(inference.artifact["pipeline"].named_steps["preprocess"], "transform", transform)
    text = "<b>GV</b> ko nhiệt tình!!!"
    result = inference.predict(text)
    transform.assert_called_once_with([text])
    assert result["error"] == "RuntimeError"
    assert result["label"] is None and result["probabilities"] is None
    assert "Suy luận thất bại" in result["status"]


def test_long_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="ký tự"):
        InferencePipeline(tmp_path / "absent.pkl").predict("a" * (MAX_TEXT_LENGTH + 1))


def test_main_comparison_keeps_all_three_models():
    from demo.app import EXAMPLES, create_app

    assert list(EXAMPLES) == ["Tích cực", "Tiêu cực", "Trung tính", "Teencode", "Không dấu"]
    assert EXAMPLES["Teencode"] == "gv dạy ko rõ ràng, sv k hiểu bài"
    config = create_app().config
    assert config["title"] == "Phân tích cảm xúc tiếng Việt"
    routes = [event for event in config["dependencies"] if event["api_visibility"] == "public"]
    assert [event["api_name"] for event in routes] == ["compare"]
    assert len(routes[0]["outputs"]) == 6
    assert sum(c["type"] == "label" for c in config["components"]) == 3
    assert sum(c["type"] == "textbox" for c in config["components"]) == 1
    for component in config["components"]:
        for key in ("label", "placeholder", "value"):
            value = component.get("props", {}).get(key)
            if isinstance(value, str):
                assert "?" not in value, (key, value)
