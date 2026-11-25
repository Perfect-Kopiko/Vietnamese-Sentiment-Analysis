"""Kiểm tra dữ liệu, tiền xử lý và kết quả baseline."""

import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from demo.inference_pipeline import InferencePipeline, MODEL_NAMES
from scripts.train_tfidf import make_scenarios
from src.datasets.tfidf_data import read_split, sha256_file
from src.evaluation.metrics import classification_metrics
from src.models.tfidf_baseline import LABELS, ROOT, TfidfPreprocessor, load_checkpoint
from src.preprocessing.noise_simulator import remove_accents


@pytest.fixture(scope="module")
def artifact():
    path = ROOT / "checkpoints/best_tfidf.pkl"
    if not path.is_file():
        pytest.skip("Run python -m scripts.train_tfidf for artifact integration tests")
    return load_checkpoint(path)


def test_normalization_and_dictionary_snapshot():
    mapping = {"ko": "không"}
    processor = TfidfPreprocessor(normalize_noise=True, fold_accents=True, mapping=mapping)
    processor.fit([])
    mapping["ko"] = "đã bị thay đổi"
    assert processor.transform(["<b>KO</b> tốt!!! https://example.com"]) == ["khong tot!"]
    assert clone(processor).get_params()["normalize_noise"] is True


def test_segmentation_really_enabled():
    processor = TfidfPreprocessor(use_segment=True).fit([])
    assert "giảng_viên" in processor.transform(["giảng viên nhiệt tình"])[0]


def test_disable_preprocessing_options():
    processor = TfidfPreprocessor().fit([])
    assert processor.transform(["GV ko tốt"]) == ["gv ko tốt"]


def test_noise_is_deterministic_and_does_not_mutate_input():
    texts = ["Không học được gì, rất khó và nhiều bài tập.", "Mình chưa nhận xét."]
    original = list(texts)
    first = make_scenarios(texts, seed=42, probability=1.0)
    second = make_scenarios(texts, seed=42, probability=1.0)
    assert first == second
    assert texts == original
    assert first["no_accents"][0] == "Khong hoc duoc gi, rat kho va nhieu bai tap."
    assert first["teencode"] != first["clean"]
    assert remove_accents("ĐẦY ĐỦ") == "DAY DU"


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_missing_checkpoint_has_no_invented_prediction(tmp_path, model):
    inference = InferencePipeline(tmp_path / "does-not-exist.pkl",
                                  bilstm_checkpoint=tmp_path / "absent.pt",
                                  phobert_checkpoint=tmp_path / "absent-phobert.pt")
    result = inference.predict("Thầy giảng dễ hiểu.", model)
    assert result["label"] is None
    assert result["probabilities"] is None
    assert "Chưa sẵn sàng" in result["status"]


def test_corrupt_checkpoint_is_reported(tmp_path):
    checkpoint = tmp_path / "bad.pkl"
    checkpoint.write_bytes(b"corrupted")
    inference = InferencePipeline(checkpoint)
    assert "Không nạp được" in inference.status(MODEL_NAMES[0])
    assert inference.predict("xin chào")["probabilities"] is None


@pytest.mark.parametrize("text", ["", "  ", None])
def test_blank_input_is_rejected(tmp_path, text):
    with pytest.raises(ValueError, match="nội dung"):
        InferencePipeline(tmp_path / "absent.pkl").predict(text)


def test_demo_matches_real_checkpoint(artifact):
    inference = InferencePipeline()
    texts = ["Thầy giảng hay và nhiệt tình.", "Môn học có ba tín chỉ.",
             "gv ko nhiệt tình, dạy khó hiểu", "thay day rat hay"]
    probabilities = artifact["pipeline"].predict_proba(texts)
    for text, expected in zip(texts, probabilities):
        result = inference.predict(text)
        actual = [result["probabilities"][LABELS[i]] for i in [0, 1, 2]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        assert sum(actual) == pytest.approx(1.0)
        assert result["label_id"] == int(artifact["pipeline"].classes_[np.argmax(expected)])


def test_checkpoint_roundtrip_and_train_only_vocabulary(artifact, tmp_path):
    path = tmp_path / "roundtrip.pkl"
    joblib.dump(artifact, path)
    reloaded = load_checkpoint(path)
    pipeline = reloaded["pipeline"]
    before = dict(pipeline.named_steps["tfidf"].vocabulary_)
    sample = ["never_seen_token_abcdef_987654321"]
    np.testing.assert_allclose(pipeline.predict_proba(sample), artifact["pipeline"].predict_proba(sample))
    assert pipeline.named_steps["tfidf"].vocabulary_ == before
    assert "never_seen_token_abcdef_987654321" not in before


def test_selected_by_validation_only(artifact):
    search = pd.read_csv(ROOT / "reports/tfidf/validation_search.csv")
    best = search.loc[search["macro_f1"].idxmax()]
    assert artifact["selection_split"] == "validation"
    assert artifact["selection_metric"] == "macro_f1"
    assert artifact["selection"]["experiment"] == best["experiment"]
    assert artifact["selection"]["C"] == best["C"]
    assert artifact["selection"]["macro_f1"] == pytest.approx(best["macro_f1"])


def test_original_data_hashes_and_labels(artifact):
    manifest = json.loads((ROOT / "data/raw/manifest.json").read_text(encoding="utf-8"))
    for split, expected in artifact["data_sha256"].items():
        path = ROOT / f"data/raw/{split}.csv"
        frame = read_split(path)
        assert sha256_file(path) == expected == manifest["splits"][split]["sha256"]
        assert len(frame) == manifest["splits"][split]["rows"]


def test_saved_metrics_match_real_predictions(artifact):
    results = pd.read_csv(ROOT / "reports/tfidf/experiment_results.csv")
    selected = results[(results["split"] == "test") & results["selected_for_demo"]]
    test = read_split(ROOT / "data/raw/test.csv")
    assert len(selected) == 3
    scenarios = make_scenarios(test["text"].tolist(), 42, 0.3)
    for row in selected.itertuples():
        saved = pd.read_csv(ROOT / f"reports/tfidf/predictions_test_{row.scenario}.csv")
        probabilities = artifact["pipeline"].predict_proba(scenarios[row.scenario])
        np.testing.assert_allclose(saved[[f"probability_{i}" for i in range(3)]], probabilities, atol=1e-12)
        assert saved["label"].tolist() == test["label"].tolist()
        assert saved["prediction"].tolist() == probabilities.argmax(axis=1).tolist()
        metrics = classification_metrics(saved["label"], saved["prediction"])
        assert row.accuracy == pytest.approx(metrics["accuracy"])
        assert row.macro_f1 == pytest.approx(metrics["macro_f1"])
