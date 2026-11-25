"""Suy luận BiLSTM trên CPU từ checkpoint đã huấn luyện."""

import torch

from src.datasets.vocab import Vocabulary
from src.models.bilstm import BiLSTMSentiment
from src.preprocessing import preprocess_pipeline


class BiLSTMInference:
    def __init__(self, checkpoint):
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.data_config = artifact["data_config"]
        self.max_len = int(self.data_config["max_len"])
        if self.max_len < 1:
            raise ValueError("Invalid BiLSTM max_len")
        self.vocab = Vocabulary()
        self.vocab.word2idx = dict(artifact["vocab_word2idx"])
        mapping = self.vocab.word2idx
        if (mapping.get("<PAD>") != 0 or mapping.get("<UNK>") != 1
                or set(mapping.values()) != set(range(len(mapping)))):
            raise ValueError("Invalid BiLSTM vocabulary")
        self.vocab.idx2word = {index: word for word, index in mapping.items()}
        state = artifact["model_state_dict"]
        vocab_size, embed_dim = state["embedding.weight"].shape
        if vocab_size != len(self.vocab) or artifact["model_config"]["num_classes"] != 3:
            raise ValueError("Invalid BiLSTM embedding or class mapping")
        self.model = BiLSTMSentiment(
            vocab_size=vocab_size, embed_dim=embed_dim, **artifact["model_config"]
        ).cpu()
        # Embedding đã nằm trong state_dict, không cần nạp file Word2Vec gốc.
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def encode(self, text, normalize_noise=None):
        if normalize_noise is None:
            normalize_noise = self.data_config["normalize_noise"]
        processed = preprocess_pipeline(
            text, use_segment=self.data_config["use_word_segment"],
            normalize_noise=normalize_noise,
        )
        tokens = self.vocab.encode(processed)[:self.max_len]
        padding = self.max_len - len(tokens)
        return (
            torch.tensor([tokens + [Vocabulary.PAD_IDX] * padding], dtype=torch.long),
            torch.tensor([[1] * len(tokens) + [0] * padding], dtype=torch.long),
        )

    def predict_proba(self, text, normalize_noise=None):
        input_ids, mask = self.encode(text, normalize_noise)
        with torch.inference_mode():
            return self.model(input_ids, mask).softmax(dim=-1)[0].numpy()
