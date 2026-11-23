import os
import sys
import json
import random
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Thêm thư mục gốc vào sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.preprocessing import preprocess_pipeline
from src.models.phobert_classifier import PhoBERTClassifier
from src.trainers.phobert_trainer import SentimentDataset, PhoBERTTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhoBERT cho UIT-VSFC")
    parser.add_argument("--config", type=str, default="configs/phobert.yaml")
    parser.add_argument("--use_segment", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--normalize_noise", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--use_class_weights", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Chạy thử nhanh với 32 mẫu trên CPU")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ghi đè tham số nếu truyền từ CLI (phục vụ Ablation)
    if args.use_segment is not None:
        config["data"]["use_segment"] = args.use_segment == "true"
    if args.normalize_noise is not None:
        config["data"]["normalize_noise"] = args.normalize_noise == "true"
    if args.use_class_weights is not None:
        config["training"]["use_class_weights"] = args.use_class_weights == "true"
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.save_dir is not None:
        config["training"]["save_dir"] = args.save_dir
    if args.run_name is not None:
        config["logging"]["wandb_run_name"] = args.run_name

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị chạy: {device}")

    # 1. Đọc dữ liệu
    data_cfg = config["data"]
    train_df = pd.read_csv(data_cfg["train_path"])
    val_df = pd.read_csv(data_cfg["val_path"])
    test_df = pd.read_csv(data_cfg["test_path"])

    if args.debug:
        print(">>> CHẾ ĐỘ DEBUG: Chỉ lấy 32 mẫu train, 16 mẫu val/test và chạy 1 epoch.")
        train_df = train_df.head(32)
        val_df = val_df.head(16)
        test_df = test_df.head(16)
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 8
        config["training"]["save_dir"] = "checkpoints/debug_phobert"

    # 2. Tiền xử lý bằng pipeline chung của nhóm
    use_seg = data_cfg.get("use_segment", True)
    norm_noise = data_cfg.get("normalize_noise", True)
    print(f"Tiền xử lý: use_segment={use_seg}, normalize_noise={norm_noise}")

    text_col = data_cfg["text_col"]
    label_col = data_cfg["label_col"]

    for df in (train_df, val_df, test_df):
        df[text_col] = df[text_col].apply(
            lambda x: preprocess_pipeline(x, use_segment=use_seg, normalize_noise=norm_noise)
        )

    # 3. Khởi tạo Tokenizer & DataLoader
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_name"])
    max_len = data_cfg.get("max_length", 128)
    bs = config["training"]["batch_size"]

    train_ds = SentimentDataset(train_df[text_col], train_df[label_col], tokenizer, max_len)
    val_ds = SentimentDataset(val_df[text_col], val_df[label_col], tokenizer, max_len)
    test_ds = SentimentDataset(test_df[text_col], test_df[label_col], tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    # 4. Khởi tạo Model & Trainer
    model = PhoBERTClassifier(
        pretrained_name=model_cfg["pretrained_name"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.3),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
    )
    trainer = PhoBERTTrainer(model, tokenizer, config, device)

    # 5. Huấn luyện
    trainer.fit(train_loader, val_loader, train_df[label_col].values)

    # 6. Load lại best checkpoint và đánh giá trên tập Test
    save_dir = config["training"]["save_dir"]
    best_model_path = os.path.join(save_dir, "model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = trainer.evaluate(test_loader)
        print("\n=== KẾT QUẢ TRÊN TẬP TEST ===")
        print(f"Accuracy   : {test_metrics['accuracy']:.4f}")
        print(f"Macro-F1   : {test_metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {test_metrics['weighted_f1']:.4f}")
        print(test_metrics["report"])

        # Lưu kết quả test ra file JSON để tổng hợp bảng Ablation
        with open(os.path.join(save_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_name": config["logging"]["wandb_run_name"],
                    "use_segment": use_seg,
                    "normalize_noise": norm_noise,
                    "use_class_weights": config["training"]["use_class_weights"],
                    "accuracy": test_metrics["accuracy"],
                    "macro_f1": test_metrics["macro_f1"],
                    "weighted_f1": test_metrics["weighted_f1"],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


if __name__ == "__main__":
    main()