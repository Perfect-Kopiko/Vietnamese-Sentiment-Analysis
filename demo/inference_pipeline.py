"""Nạp checkpoint TF-IDF và dự đoán cho demo."""

import logging
from pathlib import Path

import numpy as np

from src.models.tfidf_baseline import LABELS, ROOT, load_checkpoint

MODEL_NAMES = ("TF-IDF + Logistic Regression", "BiLSTM", "PhoBERT")
MAX_TEXT_LENGTH = 10000
LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, checkpoint=None):
        self.checkpoint = Path(checkpoint) if checkpoint else ROOT / "checkpoints/best_tfidf.pkl"
        self.artifact = None
        self.load_error = None
        if self.checkpoint.is_file():
            try:
                self.artifact = load_checkpoint(self.checkpoint)
            except Exception as exc:
                self.load_error = f"Không nạp được checkpoint: {type(exc).__name__}: {exc}"

    def status(self, model):
        if model == "BiLSTM":
            has_checkpoint = any(path.is_file() for path in (ROOT / "checkpoints/bilstm").glob("*_best.pt"))
            return ("Chưa sẵn sàng. Đã có checkpoint, còn thiếu mã suy luận."
                    if has_checkpoint else "Chưa sẵn sàng. Thiếu checkpoint BiLSTM.")
        if model == "PhoBERT":
            has_checkpoint = (ROOT / "checkpoints/best_phobert/model.pt").is_file()
            return ("Chưa sẵn sàng. Đã có checkpoint, còn thiếu mã suy luận offline."
                    if has_checkpoint else "Chưa sẵn sàng. Thiếu checkpoint PhoBERT.")
        if self.load_error:
            return self.load_error
        if self.artifact is None:
            return f"Chưa sẵn sàng. Thiếu {self.checkpoint.name}."
        return f"Sẵn sàng · {self.checkpoint.name}"

    def predict(self, text, model=MODEL_NAMES[0]):
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Vui lòng nhập nội dung bình luận.")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Bình luận tối đa {MAX_TEXT_LENGTH:,} ký tự.")
        result = {"model": model, "status": self.status(model), "label": None,
                  "label_id": None, "probabilities": None, "error": None}
        if model != MODEL_NAMES[0] or self.artifact is None:
            return result
        pipeline = self.artifact["pipeline"]
        try:
            # Truyền văn bản gốc để pipeline tự tiền xử lý như lúc huấn luyện.
            probabilities = np.asarray(pipeline.predict_proba([text])[0], dtype=float)
            classes = np.asarray(pipeline.classes_)
            if (classes.shape != (3,) or set(classes.tolist()) != set(LABELS)
                    or probabilities.shape != (3,) or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.isclose(probabilities.sum(), 1.0)):
                raise ValueError("Invalid class mapping or probabilities")
            label_id = int(classes[probabilities.argmax()])
            result.update(label_id=label_id, label=LABELS[label_id],
                          probabilities={LABELS[int(label)]: float(probability)
                                         for label, probability in zip(classes, probabilities)})
        except Exception as exc:
            LOGGER.exception("TF-IDF inference failed")
            result.update(status=f"Suy luận thất bại ({type(exc).__name__}).",
                          error=type(exc).__name__)
        return result

    def compare(self, text):
        return [self.predict(text, model) for model in MODEL_NAMES]
