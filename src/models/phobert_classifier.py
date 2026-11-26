# Kiến trúc PhoBERT Classifier fine-tuning
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class PhoBERTClassifier(nn.Module):
    """
    Mô hình phân loại cảm xúc 3 lớp dựa trên PhoBERT.
    Hỗ trợ trả về attention weights để phục vụ trực quan hóa lỗi.
    """

    def __init__(
        self,
        pretrained_name: str = "vinai/phobert-base-v2",
        num_classes: int = 3,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        *,
        backbone_config=None,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name
        self.num_classes = num_classes

        if backbone_config is None:
            self.config = AutoConfig.from_pretrained(pretrained_name)
            self.backbone = AutoModel.from_pretrained(
                pretrained_name, config=self.config, attn_implementation="eager",
            )
        else:
            # Demo dựng kiến trúc cục bộ rồi nạp toàn bộ state_dict đã fine-tune.
            self.config = backbone_config
            self.backbone = AutoModel.from_config(self.config, attn_implementation="eager")

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True,
        )

        # Lấy vector đại diện của token đầu câu (<s> / CLS)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        if output_attentions:
            return logits, outputs.attentions
        return logits


if __name__ == "__main__":
    # Test nhanh kiến trúc model trên CPU
    print("Đang khởi tạo thử PhoBERTClassifier...")
    model = PhoBERTClassifier(pretrained_name="vinai/phobert-base-v2", num_classes=3)
    dummy_ids = torch.randint(0, 1000, (2, 16))
    dummy_mask = torch.ones((2, 16), dtype=torch.long)

    logits, attentions = model(dummy_ids, dummy_mask, output_attentions=True)
    print("Logits shape:", logits.shape)             # Kỳ vọng: [2, 3]
    print("Số tầng Attention:", len(attentions))     # Kỳ vọng: 12
    print("Attention tầng cuối:", attentions[-1].shape)
