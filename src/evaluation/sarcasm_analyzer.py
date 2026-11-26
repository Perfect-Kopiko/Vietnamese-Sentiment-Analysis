# Phân tích lỗi cho các câu mỉa mai và phủ định kép
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

# Đảm bảo import được các module từ thư mục gốc
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.preprocessing import preprocess_pipeline
from src.models.phobert_classifier import PhoBERTClassifier

LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}
STR_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}


def load_sarcasm_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Chuẩn hóa cột sentiment về dạng số (0, 1, 2) nếu đang là chữ
    if df["sentiment"].dtype == object:
        df["label"] = df["sentiment"].str.strip().str.lower().map(STR_TO_INT)
    else:
        df["label"] = df["sentiment"].astype(int)
    return df


@torch.no_grad()
def predict_phobert(
    texts,
    checkpoint_dir: str = "checkpoints/best_phobert",
    pretrained_name: str = "vinai/phobert-base-v2",
    max_length: int = 128,
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = PhoBERTClassifier(pretrained_name=pretrained_name, num_classes=3)
    model_path = os.path.join(checkpoint_dir, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    processed_texts = [
        preprocess_pipeline(t, use_segment=True, normalize_noise=True) for t in texts
    ]

    preds, confidences = [], []
    for text in processed_texts:
        enc = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        preds.append(pred)
        confidences.append(float(probs[pred]))

    return np.array(preds), np.array(confidences)


def compute_sarcasm_metrics(model_name: str, df: pd.DataFrame, preds: np.ndarray) -> dict:
    y_true = df["label"].values
    overall_acc = accuracy_score(y_true, preds)
    overall_err = 1.0 - overall_acc
    macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    # Tách riêng nhóm Mỉa mai (is_sarcastic=1) và Phủ định kép (is_sarcastic=0)
    sarc_mask = df["is_sarcastic"] == 1
    neg_mask = df["is_sarcastic"] == 0

    sarc_err = 1.0 - accuracy_score(y_true[sarc_mask], preds[sarc_mask]) if sarc_mask.any() else np.nan
    neg_err = 1.0 - accuracy_score(y_true[neg_mask], preds[neg_mask]) if neg_mask.any() else np.nan

    return {
        "Model": model_name,
        "Total Samples": len(df),
        "Accuracy (%)": round(overall_acc * 100, 2),
        "Error Rate (%)": round(overall_err * 100, 2),
        "Sarcasm Error (%)": round(sarc_err * 100, 2),
        "Double-Neg Error (%)": round(neg_err * 100, 2),
        "Macro-F1": round(macro_f1, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Phân tích lỗi mỉa mai / phủ định kép")
    parser.add_argument("--data_path", type=str, default="data/resources/sarcasm_test.csv")
    parser.add_argument("--phobert_dir", type=str, default="checkpoints/best_phobert")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_sarcasm_dataset(args.data_path)
    print(f"Đã tải {len(df)} câu từ {args.data_path}")

    summary_rows = []

    # 1. Đánh giá PhoBERT
    if os.path.exists(os.path.join(args.phobert_dir, "model.pt")):
        print("\nĐang đánh giá PhoBERT...")
        preds, confs = predict_phobert(df["text"].tolist(), checkpoint_dir=args.phobert_dir)
        summary_rows.append(compute_sarcasm_metrics("PhoBERT-v2", df, preds))

        # Xuất danh sách câu đoán sai của PhoBERT để phân tích trong báo cáo
        error_df = df.copy()
        error_df["pred"] = preds
        error_df["true_label"] = error_df["label"].map(LABEL_MAP)
        error_df["pred_label"] = error_df["pred"].map(LABEL_MAP)
        error_df["confidence"] = np.round(confs, 4)
        error_df = error_df[error_df["label"] != error_df["pred"]]

        err_path = os.path.join(args.output_dir, "sarcasm_phobert_errors.csv")
        error_df[["text", "true_label", "pred_label", "confidence", "is_sarcastic", "notes"]].to_csv(
            err_path, index=False, encoding="utf-8-sig"
        )
        print(f"PhoBERT đoán sai {len(error_df)}/{len(df)} câu. Đã lưu tại: {err_path}")
    else:
        print(f"[Cảnh báo] Chưa tìm thấy checkpoint PhoBERT tại {args.phobert_dir}/model.pt")

    # 2. (Mở rộng sau) Khi 2 bạn cùng nhóm hoàn thành TF-IDF và BiLSTM,
    #    chỉ cần gọi hàm predict của 2 model đó và append vào summary_rows:
    #    summary_rows.append(compute_sarcasm_metrics("TF-IDF + LR", df, tfidf_preds))
    #    summary_rows.append(compute_sarcasm_metrics("BiLSTM", df, bilstm_preds))

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n=== BẢNG SO SÁNH TỶ LỆ ĐOÁN SAI TRÊN TẬP SARCASM ===")
        print(summary_df.to_string(index=False))

        out_csv = os.path.join(args.output_dir, "sarcasm_comparison.csv")
        summary_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\nĐã lưu bảng tổng hợp tại: {out_csv}")


if __name__ == "__main__":
    main()