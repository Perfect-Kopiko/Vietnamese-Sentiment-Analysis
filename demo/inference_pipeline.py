"""Nạp checkpoint cục bộ và dự đoán cho demo."""

from copy import copy
import logging
from pathlib import Path

import numpy as np

from src.models.tfidf_baseline import LABELS, ROOT, load_checkpoint

MODEL_NAMES = ("TF-IDF + Logistic Regression", "BiLSTM", "PhoBERT")
MAX_TEXT_LENGTH = 10000
LOGGER = logging.getLogger(__name__)


class InferencePipeline:
    def __init__(self, checkpoint=None, *, bilstm_checkpoint=None, phobert_checkpoint=None):
        self.checkpoint = Path(checkpoint) if checkpoint else ROOT / "checkpoints/best_tfidf.pkl"
        candidates = (ROOT / "checkpoints/bilstm/bilstm_no_seg_w2v_best.pt",
                      ROOT / "bilstm_no_seg_w2v_best.pt")
        self.bilstm_checkpoint = (Path(bilstm_checkpoint) if bilstm_checkpoint is not None
                                  else next((p for p in candidates if p.is_file()), candidates[0]))
        self.phobert_checkpoint = (Path(phobert_checkpoint) if phobert_checkpoint is not None
                                   else ROOT / "checkpoints/best_phobert/model.pt")
        self.artifact = None
        self.load_error = None
        self.bilstm = None
        self.bilstm_error = None
        if self.checkpoint.is_file():
            try:
                self.artifact = load_checkpoint(self.checkpoint)
            except Exception as exc:
                self.load_error = f"Không nạp được checkpoint: {type(exc).__name__}: {exc}"
        if self.bilstm_checkpoint.is_file():
            try:
                from demo.bilstm_inference import BiLSTMInference

                self.bilstm = BiLSTMInference(self.bilstm_checkpoint)
            except Exception as exc:
                LOGGER.exception("BiLSTM loading failed")
                self.bilstm_error = f"Không nạp được BiLSTM: {type(exc).__name__}: {exc}"

    def status(self, model):
        if model == "BiLSTM":
            if self.bilstm_error:
                return self.bilstm_error
            return (f"Sẵn sàng · {self.bilstm_checkpoint.name} · CPU" if self.bilstm is not None
                    else f"Chưa sẵn sàng. Thiếu checkpoint BiLSTM: {self.bilstm_checkpoint.name}.")
        if model == "PhoBERT":
            return ("Chưa sẵn sàng. Đã có checkpoint PhoBERT, còn thiếu kiến trúc, tokenizer "
                    "và cấu hình suy luận từ nhánh PhoBERT."
                    if self.phobert_checkpoint.is_file() else
                    "Chưa sẵn sàng. Thiếu checkpoint PhoBERT và mã suy luận trên main.")
        if self.load_error:
            return self.load_error
        if self.artifact is None:
            return f"Chưa sẵn sàng. Thiếu {self.checkpoint.name}."
        return f"Sẵn sàng · {self.checkpoint.name}"

    def predict(self, text, model=MODEL_NAMES[0], normalize_noise=False):
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model: {model}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Vui lòng nhập nội dung bình luận.")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Bình luận tối đa {MAX_TEXT_LENGTH:,} ký tự.")
        result = {"model": model, "status": self.status(model), "label": None,
                  "label_id": None, "probabilities": None, "error": None}
        if model == "PhoBERT" or (model == "BiLSTM" and self.bilstm is None):
            return result
        if model == MODEL_NAMES[0] and self.artifact is None:
            return result
        try:
            if model == "BiLSTM":
                probabilities = np.asarray(self.bilstm.predict_proba(text, normalize_noise), dtype=float)
                classes = np.asarray([0, 1, 2])
            else:
                pipeline = self.artifact["pipeline"]
                processor = pipeline.named_steps["preprocess"]
                if processor.normalize_noise != normalize_noise:
                    # Chỉ đổi bước chuẩn hóa của lượt chạy, giữ nguyên mô hình đã fit.
                    processor = copy(processor)
                    processor.normalize_noise = normalize_noise
                    pipeline = copy(pipeline)
                    pipeline.steps = [("preprocess", processor), *pipeline.steps[1:]]
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
            LOGGER.exception("%s inference failed", model)
            result.update(status=f"Suy luận thất bại ({type(exc).__name__}).",
                          error=type(exc).__name__)
        return result

    def compare(self, text, normalize_noise=False):
        return [self.predict(text, model, normalize_noise) for model in MODEL_NAMES]
