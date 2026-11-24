"""Pipeline TF-IDF và Logistic Regression."""

import json
from pathlib import Path

import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.noise_normalizer import NoiseNormalizer
from src.preprocessing.noise_simulator import remove_accents
from src.preprocessing.text_cleaner import clean_text

ROOT = Path(__file__).resolve().parents[2]
LABELS = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}


class TfidfPreprocessor(TransformerMixin, BaseEstimator):
    """Tiền xử lý văn bản theo cấu hình baseline."""

    def __init__(self, use_segment=False, normalize_noise=False,
                 fold_accents=False, mapping=None):
        self.use_segment = use_segment
        self.normalize_noise = normalize_noise
        self.fold_accents = fold_accents
        self.mapping = mapping

    def fit(self, X, y=None):
        mapping = self.mapping
        if mapping is None:
            path = ROOT / "data/resources/teencode_dict.json"
            mapping = json.loads(path.read_text(encoding="utf-8"))
        # Lưu bản sao từ điển để mô hình không phụ thuộc vào lần sửa JSON sau này.
        self.mapping_ = dict(mapping)
        self.pattern_ = NoiseNormalizer._compile_regex(self.mapping_)
        if self.use_segment:
            from pyvi import ViTokenizer  # Báo lỗi nếu thiếu PyVi khi bật tách từ.
        return self

    def transform(self, X):
        if self.use_segment:
            from pyvi import ViTokenizer
        results = []
        for text in X:
            text = clean_text(text)
            if self.normalize_noise and self.pattern_ is not None:
                text = self.pattern_.sub(
                    lambda match: self.mapping_[match.group(0).lower()], text
                )
            if self.use_segment and text:
                text = ViTokenizer.tokenize(text)
            if self.fold_accents:
                text = remove_accents(text)
            results.append(text)
        return results


def build_pipeline(config, *, use_segment=False, normalize_noise=False,
                   fold_accents=False, C=1.0):
    params = dict(config["tfidf"])
    params["ngram_range"] = tuple(params["ngram_range"])
    # Giữ token một ký tự như k, m trong teencode.
    params.update(lowercase=False, token_pattern=r"(?u)\b\w+\b")
    lr = config["logistic_regression"]
    return Pipeline([
        ("preprocess", TfidfPreprocessor(use_segment, normalize_noise, fold_accents)),
        ("tfidf", TfidfVectorizer(**params)),
        ("classifier", LogisticRegression(
            C=C, class_weight=lr["class_weight"], solver=lr["solver"],
            max_iter=lr["max_iter"], random_state=config["seed"],
        )),
    ])


def load_checkpoint(path):
    """Nạp và kiểm tra checkpoint từ nguồn tin cậy."""
    artifact = joblib.load(path)
    if artifact.get("format_version") != 1:
        raise ValueError("Unsupported TF-IDF checkpoint format")
    if list(artifact["pipeline"].classes_) != [0, 1, 2]:
        raise ValueError("Checkpoint must contain sentiment labels 0, 1, 2")
    return artifact
