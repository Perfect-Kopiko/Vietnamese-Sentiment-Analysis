"""Suy luận PhoBERT trên CPU bằng tài nguyên cục bộ."""

import json
from pathlib import Path

import torch
from transformers import PhobertTokenizer, RobertaConfig

from src.models.phobert_classifier import PhoBERTClassifier
from src.preprocessing import preprocess_pipeline

ROOT = Path(__file__).resolve().parents[1]


class PhoBERTInference:
    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        directory = self.checkpoint.parent
        required = ("config.json", "tokenizer_config.json", "added_tokens.json",
                    "vocab.txt", "bpe.codes")
        for name in required:
            if not (directory / name).is_file():
                raise FileNotFoundError(f"Thiếu tài nguyên PhoBERT: {directory / name}")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.data_config = config["data"]
        self.max_len = int(self.data_config["max_length"])
        model_config = config["model"]
        self.class_ids = (0, 1, 2)  # Thứ tự labels trong PhoBERTTrainer.evaluate.
        if (model_config["pretrained_name"] != "vinai/phobert-base-v2"
                or model_config["num_classes"] != len(self.class_ids)):
            raise ValueError("Invalid PhoBERT backbone or class mapping")
        backbone_config = RobertaConfig.from_json_file(ROOT / "configs/phobert_backbone.json")
        if not 2 <= self.max_len <= backbone_config.max_position_embeddings - 2:
            raise ValueError("Invalid PhoBERT max_length")
        self.tokenizer = PhobertTokenizer.from_pretrained(directory, local_files_only=True)
        if (len(self.tokenizer) != backbone_config.vocab_size
                or (self.tokenizer.bos_token_id, self.tokenizer.pad_token_id,
                    self.tokenizer.eos_token_id, self.tokenizer.unk_token_id) != (0, 1, 2, 3)):
            raise ValueError("Invalid PhoBERT tokenizer")
        if self.data_config["use_segment"]:
            from pyvi import ViTokenizer  # Bắt lỗi thiếu PyVi thay vì bỏ qua tách từ.

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        # Không cấp phát thêm một bản trọng số ngẫu nhiên trong lúc nạp.
        with torch.device("meta"):
            self.model = PhoBERTClassifier(**model_config, backbone_config=backbone_config)
        self.model.load_state_dict(state, strict=True, assign=True)
        # Hai buffer RoBERTa không nằm trong state_dict, cần tạo lại trên CPU.
        embeddings = self.model.backbone.embeddings
        positions = torch.arange(backbone_config.max_position_embeddings).expand((1, -1))
        embeddings.register_buffer("position_ids", positions, persistent=False)
        embeddings.register_buffer("token_type_ids", torch.zeros_like(positions), persistent=False)
        self.model.requires_grad_(False)
        self.model.eval()

    def encode(self, text, normalize_noise=None):
        if normalize_noise is None:
            normalize_noise = self.data_config["normalize_noise"]
        processed = preprocess_pipeline(
            text, use_segment=self.data_config["use_segment"], normalize_noise=normalize_noise,
        )
        return self.tokenizer(
            processed, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt", return_token_type_ids=False,
        )

    def predict_proba(self, text, normalize_noise=None):
        inputs = self.encode(text, normalize_noise)
        with torch.inference_mode():
            return self.model(**inputs).softmax(dim=-1)[0].numpy()
