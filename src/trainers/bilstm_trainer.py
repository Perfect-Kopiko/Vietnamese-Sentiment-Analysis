"""
Trainer huấn luyện và đánh giá mô hình BiLSTM phân loại cảm xúc tiếng Việt.
Hỗ trợ:
- Huấn luyện vòng lặp chuẩn (Train/Val/Test)
- Gradient Clipping tránh bùng nổ gradient trong mạng hồi quy LSTM
- Tích hợp Weights & Biases (W&B) với cơ chế fallback offline an toàn
- Lưu checkpoint mô hình tốt nhất (Best Val Macro-F1) kèm Vocab để phục vụ inference
- Đánh giá trên tập kiểm tra (Test set) và trích xuất báo cáo JSON
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.vocab import Vocabulary
from src.evaluation.metrics import compute_metrics, format_metrics_report


class BiLSTMTrainer:
    """
    Quản lý quá trình huấn luyện, xác thực, lưu trữ mô hình BiLSTM và tích hợp W&B.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
        config: Dict[str, Any],
        vocab: Vocabulary,
        run_name: str = "bilstm_experiment"
    ):
        """
        Khởi tạo BiLSTMTrainer.

        Args:
            model: Mô hình PyTorch BiLSTMSentiment.
            train_loader: DataLoader tập train.
            val_loader: DataLoader tập validation.
            test_loader: DataLoader tập test.
            criterion: Hàm mất mát (CrossEntropy, Weighted CE hoặc Focal Loss).
            optimizer: Bộ tối ưu hóa (ví dụ Adam, AdamW).
            scheduler: Bộ điều chỉnh learning rate (ví dụ ReduceLROnPlateau).
            device: Thiết bị chạy (cuda hoặc cpu).
            config: Toàn bộ cấu hình thực nghiệm dạng dictionary.
            vocab: Đối tượng Vocabulary đã fit trên tập train.
            run_name: Tên của phiên chạy thử nghiệm.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.vocab = vocab
        self.run_name = run_name

        training_cfg = config.get("training", {})
        self.epochs = int(training_cfg.get("epochs", 15))
        self.grad_clip = float(training_cfg.get("grad_clip", 1.0))
        self.checkpoint_dir = training_cfg.get("checkpoint_dir", "checkpoints/bilstm")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.run_name}_best.pt"
        )
        self.best_val_macro_f1 = -1.0
        self.best_val_metrics: Dict[str, Any] = {}

        # Khởi tạo Weights & Biases
        self.use_wandb = bool(training_cfg.get("use_wandb", True))
        self.wandb_run = None
        if self.use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Khởi tạo W&B với fallback offline nếu mất kết nối hoặc chưa đăng nhập."""
        try:
            import wandb
            project_name = self.config.get("training", {}).get(
                "wandb_project", "vietnamese-sentiment-topic6"
            )
            # Khởi tạo run
            self.wandb_run = wandb.init(
                project=project_name,
                name=self.run_name,
                config=self.config,
                reinit=True
            )
            print(f"[+] Đã kết nối thành công Weights & Biases: {self.wandb_run.url}")
        except Exception as exc:
            print(f"[!] Không thể kết nối online với W&B ({exc}). Tự động chuyển sang chế độ OFFLINE.")
            os.environ["WANDB_MODE"] = "offline"
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.config.get("training", {}).get("wandb_project", "vietnamese-sentiment-topic6"),
                    name=self.run_name,
                    config=self.config,
                    reinit=True
                )
                print("[*] Đã bật chế độ W&B Offline thành công.")
            except Exception as e_offline:
                print(f"[!] Tắt hoàn toàn W&B do lỗi: {e_offline}")
                self.use_wandb = False
                self.wandb_run = None

    def train_epoch(self, epoch_idx: int) -> Tuple[float, Dict[str, Any]]:
        """Huấn luyện 1 epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch_idx}/{self.epochs} [Train]",
            leave=False
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)

            loss.backward()

            # Gradient Clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(self.train_loader)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        desc: str = "Evaluating"
    ) -> Tuple[float, Dict[str, Any], np.ndarray, List[int], List[int]]:
        """
        Đánh giá mô hình trên một dataloader.

        Returns:
            Tuple: (avg_loss, metrics_dict, all_probs, all_preds, all_labels)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs_list: List[np.ndarray] = []

        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_probs_list.append(probs.detach().cpu().numpy())
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        metrics = compute_metrics(all_labels, all_preds)
        all_probs = np.concatenate(all_probs_list, axis=0) if all_probs_list else np.empty(0)

        return avg_loss, metrics, all_probs, all_preds, all_labels

    def fit(self) -> Dict[str, Any]:
        """
        Vòng lặp huấn luyện chính qua các epoch.
        """
        print(f"\n[*] Bắt đầu huấn luyện mô hình: {self.run_name}")
        print(f"[*] Thiết bị: {self.device} | Tổng số epochs: {self.epochs}")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            t_epoch_start = time.time()

            # 1. Train epoch
            train_loss, train_metrics = self.train_epoch(epoch)

            # 2. Validation epoch
            val_loss, val_metrics, _, _, _ = self.evaluate(
                self.val_loader, desc=f"Epoch {epoch}/{self.epochs} [Val]"
            )

            # 3. Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["macro_f1"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - t_epoch_start

            print(
                f"Epoch {epoch:02d}/{self.epochs:02d} [{epoch_time:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics['accuracy'] * 100:.2f}% | "
                f"Val Macro-F1: {val_metrics['macro_f1'] * 100:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

            # 4. Ghi log W&B
            if self.use_wandb and self.wandb_run is not None:
                try:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/accuracy": train_metrics["accuracy"],
                        "train/macro_f1": train_metrics["macro_f1"],
                        "val/loss": val_loss,
                        "val/accuracy": val_metrics["accuracy"],
                        "val/macro_f1": val_metrics["macro_f1"],
                        "val/f1_neutral": val_metrics["f1_neutral"],
                        "lr": current_lr
                    })
                except Exception:
                    pass

            # 5. Lưu checkpoint khi đạt kỷ lục Best Val Macro-F1
            if val_metrics["macro_f1"] > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_metrics["macro_f1"]
                self.best_val_metrics = val_metrics

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "model_config": self.config.get("model", {}),
                    "data_config": self.config.get("data", {}),
                    "embedding_config": self.config.get("embedding", {}),
                    "vocab_word2idx": self.vocab.word2idx,
                    "best_val_metrics": self.best_val_metrics,
                    "config": self.config
                }
                torch.save(checkpoint, self.best_checkpoint_path)
                print(
                    f"  [+] Đạt kỷ lục Val Macro-F1 mới ({self.best_val_macro_f1 * 100:.2f}%)! "
                    f"Đã lưu checkpoint: {self.best_checkpoint_path}"
                )

        total_training_time = time.time() - start_time
        print(f"\n[+] Hoàn tất huấn luyện trong {total_training_time:.2f} giây.")
        print(f"[+] Kỷ lục Best Val Macro-F1: {self.best_val_macro_f1 * 100:.2f}%")

        return {
            "best_val_macro_f1": self.best_val_macro_f1,
            "best_val_metrics": self.best_val_metrics,
            "training_time_sec": total_training_time
        }

    def test_and_save(self, result_path: str) -> Dict[str, Any]:
        """
        Nạp lại checkpoint tốt nhất, đánh giá trên tập kiểm tra (test set),
        in báo cáo chi tiết và xuất kết quả ra file JSON.

        Args:
            result_path: Đường dẫn lưu file kết quả JSON.

        Returns:
            dict: Tổng hợp toàn bộ kết quả thực nghiệm.
        """
        if os.path.exists(self.best_checkpoint_path):
            print(f"\n[*] Đang nạp checkpoint tốt nhất từ: {self.best_checkpoint_path}")
            checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"[!] Không tìm thấy checkpoint {self.best_checkpoint_path}. Sử dụng trọng số hiện tại.")

        # Đánh giá trên tập test
        test_loss, test_metrics, _, _, _ = self.evaluate(
            self.test_loader, desc="Testing on Test Set"
        )

        report_str = format_metrics_report(
            test_metrics,
            title=f"KẾT QUẢ KIỂM THỬ TRÊN TẬP TEST ({self.run_name})"
        )
        print(report_str)

        # Log test metrics lên W&B
        if self.use_wandb and self.wandb_run is not None:
            try:
                import wandb
                wandb.log({
                    "test/loss": test_loss,
                    "test/accuracy": test_metrics["accuracy"],
                    "test/macro_f1": test_metrics["macro_f1"],
                    "test/weighted_f1": test_metrics["weighted_f1"],
                    "test/f1_negative": test_metrics["f1_negative"],
                    "test/f1_neutral": test_metrics["f1_neutral"],
                    "test/f1_positive": test_metrics["f1_positive"]
                })
                wandb.finish()
            except Exception:
                pass

        # Tổng hợp kết quả để lưu ra file JSON
        os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
        final_summary = {
            "run_name": self.run_name,
            "config": self.config,
            "best_val_macro_f1": self.best_val_macro_f1,
            "val_metrics": self.best_val_metrics,
            "test_metrics": test_metrics,
            "checkpoint_path": self.best_checkpoint_path
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

        print(f"[+] Đã lưu báo cáo kết quả thực nghiệm tại: {result_path}")
        return final_summary
