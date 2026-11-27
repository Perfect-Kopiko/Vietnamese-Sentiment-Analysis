import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class PhoBERTTrainer:
    def __init__(self, model, tokenizer, config, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.use_wandb = config["logging"].get("use_wandb", False) and (wandb is not None)

    def _build_criterion(self, train_labels):
        if self.config["training"].get("use_class_weights", True):
            classes = np.unique(train_labels)
            weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=train_labels
            )
            weight_tensor = torch.tensor(weights, dtype=torch.float).to(self.device)
            print(f"Sử dụng class weights: {np.round(weights, 4)}")
            return nn.CrossEntropyLoss(weight=weight_tensor)
        return nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, optimizer, scheduler, criterion):
        self.model.train()
        total_loss = 0.0
        preds_all, labels_all = [], []

        for batch in tqdm(dataloader, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config["training"].get("max_grad_norm", 1.0)
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(dataloader), 1)
        acc = accuracy_score(labels_all, preds_all)
        macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        return avg_loss, acc, macro_f1

    @torch.no_grad()
    def evaluate(self, dataloader, criterion=None):
        self.model.eval()
        total_loss = 0.0
        preds_all, labels_all = [], []

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(len(dataloader), 1) if criterion is not None else 0.0
        acc = accuracy_score(labels_all, preds_all)
        macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels_all, preds_all, average="weighted", zero_division=0)
        report = classification_report(
            labels_all,
            preds_all,
            labels=[0, 1, 2],  # Thêm dòng này để luôn đủ 3 lớp
            target_names=["Negative", "Neutral", "Positive"],
            digits=4,
            zero_division=0,
        )
        return {
            "loss": avg_loss,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "report": report,
            "preds": preds_all,
            "labels": labels_all,
        }

    def save_checkpoint(self, save_dir, metrics):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        self.tokenizer.save_pretrained(save_dir)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        with open(os.path.join(save_dir, "best_val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def fit(self, train_loader, val_loader, train_labels):
        train_cfg = self.config["training"]
        epochs = train_cfg["epochs"]
        save_dir = train_cfg["save_dir"]
        patience = train_cfg.get("early_stopping_patience", 2)

        criterion = self._build_criterion(train_labels)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        )

        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.1)))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        if self.use_wandb:
            wandb.init(
                project=self.config["logging"].get("wandb_project", "vietnamese-sentiment-analysis"),
                name=self.config["logging"].get("wandb_run_name", "phobert-run"),
                mode=self.config["logging"].get("wandb_mode", "offline"),
                config=self.config,
            )

        best_val_f1 = -1.0
        no_improve_epochs = 0

        for epoch in range(1, epochs + 1):
            print(f"\n--- Epoch {epoch}/{epochs} ---")
            train_loss, train_acc, train_f1 = self.train_epoch(
                train_loader, optimizer, scheduler, criterion
            )
            val_metrics = self.evaluate(val_loader, criterion)

            print(
                f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Macro-F1: {train_f1:.4f}\n"
                f"Val   | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} "
                f"| Macro-F1: {val_metrics['macro_f1']:.4f}"
            )

            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/macro_f1": train_f1,
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/macro_f1": val_metrics["macro_f1"],
                })

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                no_improve_epochs = 0
                self.save_checkpoint(save_dir, val_metrics)
                print(f"--> Đã lưu best checkpoint tại {save_dir} (Macro-F1: {best_val_f1:.4f})")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print(f"Early stopping sau {epoch} epochs.")
                    break

        if self.use_wandb:
            wandb.finish()

        return best_val_f1