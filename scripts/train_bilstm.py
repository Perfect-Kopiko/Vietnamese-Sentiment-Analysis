"""
Script huấn luyện mô hình BiLSTM (Đề tài 6: Phân tích cảm xúc văn bản tiếng Việt).
Cho phép truyền tham số linh hoạt từ dòng lệnh (argparse) phục vụ các thí nghiệm Ablation Study:
- So sánh cơ chế Pooling (attention vs mean_max)
- So sánh Hàm mất mát (ce, weighted_ce, focal)
- So sánh Tách từ ghép (use_word_segment) và Chuẩn hóa teencode (normalize_noise)
- So sánh Word2Vec pre-trained vs ngẫu nhiên
"""

import argparse
import os
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Thêm đường dẫn project root vào sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.datasets import create_dataloaders, load_word2vec_embeddings
from src.models import BiLSTMSentiment, compute_class_weights, get_loss_function
from src.trainers import BiLSTMTrainer


def set_seed(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên để đảm bảo tính tái lập (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def str2bool(v: Any) -> bool:
    """Hỗ trợ ép kiểu boolean từ tham số dòng lệnh."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình BiLSTM phân tích cảm xúc tiếng Việt.")

    parser.add_argument("--config", type=str, default="configs/bilstm.yaml", help="Đường dẫn file cấu hình YAML.")
    parser.add_argument("--run_name", type=str, default="bilstm_seg_weighted_ce", help="Tên phiên chạy thực nghiệm.")
    parser.add_argument("--pooling", type=str, choices=["attention", "mean_max"], default=None, help="Cơ chế pooling.")
    parser.add_argument("--loss_type", type=str, choices=["ce", "weighted_ce", "focal"], default=None, help="Hàm mất mát.")
    parser.add_argument("--use_word_segment", type=str2bool, default=None, help="Kích hoạt tách từ ghép tiếng Việt (true/false).")
    parser.add_argument("--normalize_noise", type=str2bool, default=None, help="Kích hoạt chuẩn hóa teencode (true/false).")
    parser.add_argument("--w2v_path", type=str, default=None, help="Đường dẫn file Word2Vec (.vec / .txt).")
    parser.add_argument("--epochs", type=int, default=None, help="Số epochs huấn luyện.")
    parser.add_argument("--batch_size", type=int, default=None, help="Kích thước batch.")
    parser.add_argument("--lr", type=float, default=None, help="Tốc độ học (Learning rate).")
    parser.add_argument("--no_wandb", action="store_true", help="Tắt hoàn toàn Weights & Biases (dùng cho test nhanh).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed tái lập kết quả.")

    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    set_seed(args.seed)

    # 1. Đọc file config YAML
    config_path = os.path.join(_project_root, args.config)
    if not os.path.exists(config_path):
        config_path = args.config
    config = load_yaml_config(config_path)

    # 2. Ghi đè cấu hình từ argparse nếu có
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = "bilstm_experiment"

    if args.pooling is not None:
        config["model"]["pooling"] = args.pooling
    if args.loss_type is not None:
        config["training"]["loss_type"] = args.loss_type
    if args.use_word_segment is not None:
        config["data"]["use_word_segment"] = args.use_word_segment
    if args.normalize_noise is not None:
        config["data"]["normalize_noise"] = args.normalize_noise
    if args.w2v_path is not None:
        config["embedding"]["w2v_path"] = args.w2v_path
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.no_wandb:
        config["training"]["use_wandb"] = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"[*] KHỞI ĐỘNG THỰC NGHIỆM: {run_name}")
    print(f"[*] Thiết bị sử dụng: {device}")
    print(f"[*] Cấu hình: {config}")
    print("=" * 70)

    # 3. Tạo DataLoaders và Vocabulary
    train_csv = os.path.join(_project_root, config["data"]["train_path"])
    val_csv = os.path.join(_project_root, config["data"]["val_path"])
    test_csv = os.path.join(_project_root, config["data"]["test_path"])

    train_loader, val_loader, test_loader, vocab = create_dataloaders(
        train_path=train_csv,
        val_path=val_csv,
        test_path=test_csv,
        batch_size=int(config["training"]["batch_size"]),
        max_len=int(config["data"]["max_len"]),
        min_freq=int(config["data"]["min_freq"]),
        use_word_segment=bool(config["data"]["use_word_segment"]),
        normalize_noise=bool(config["data"]["normalize_noise"])
    )

    # 4. Nạp Embeddings (Word2Vec hoặc ngẫu nhiên)
    w2v_path = config["embedding"].get("w2v_path", "")
    embed_dim = int(config["embedding"].get("embed_dim", 300))
    pretrained_embeds = None
    if w2v_path:
        w2v_full_path = os.path.join(_project_root, w2v_path) if not os.path.isabs(w2v_path) else w2v_path
        pretrained_embeds = load_word2vec_embeddings(
            vocab=vocab,
            w2v_path=w2v_full_path,
            embed_dim=embed_dim
        )

    # 5. Khởi tạo mô hình BiLSTMSentiment
    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        num_classes=int(config["model"]["num_classes"]),
        dropout=float(config["model"]["dropout"]),
        pooling=config["model"]["pooling"],
        pretrained_embeddings=pretrained_embeds,
        freeze_embeddings=bool(config["embedding"].get("freeze", False))
    )

    # 6. Tính Class Weights và Hàm mất mát
    # Lấy toàn bộ nhãn tập train để tính trọng số nghịch đảo tần suất
    train_labels = [batch["label"] for batch in train_loader]
    train_labels = torch.cat(train_labels, dim=0).numpy()
    class_weights = compute_class_weights(train_labels, num_classes=int(config["model"]["num_classes"]))

    loss_fn = get_loss_function(
        loss_type=config["training"]["loss_type"],
        class_weights=class_weights,
        focal_gamma=float(config["training"].get("focal_gamma", 2.0)),
        device=str(device)
    )

    # 7. Optimizer và LR Scheduler
    lr = float(config["training"]["lr"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    # 8. Trainer
    trainer = BiLSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        vocab=vocab,
        run_name=run_name
    )

    # 9. Huấn luyện
    trainer.fit()

    # 10. Đánh giá trên Test set và xuất báo cáo JSON
    results_dir = os.path.join(_project_root, "results")
    result_json_path = os.path.join(results_dir, f"{run_name}.json")
    trainer.test_and_save(result_json_path)


if __name__ == "__main__":
    main()
