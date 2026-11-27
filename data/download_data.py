"""Tải các tập dữ liệu UIT-VSFC gốc."""

import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_NAME = "uitnlp/vietnamese_students_feedback"
SOURCE_REF = "refs/convert/parquet"
# Dùng đúng phiên bản dữ liệu của baseline.
REVISION = "2d76906157232f87e9883fa37ebfc91ffdee1f83"
ROOT = Path(__file__).resolve().parents[1]
LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}


def download_and_save_data(output_dir=None):
    output_dir = Path(output_dir) if output_dir else ROOT / "data/raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(DATASET_NAME, revision=REVISION)
    print(f"Downloading {DATASET_NAME} @ {info.sha}", flush=True)
    # Giữ đường dẫn cache ngắn để tên file khóa không vượt giới hạn của Windows.
    dataset = load_dataset(DATASET_NAME, revision=info.sha, cache_dir=str(ROOT / ".cache"))
    manifest = {"dataset": DATASET_NAME, "revision": info.sha,
                "source_ref": SOURCE_REF, "labels": LABEL_MAPPING, "splits": {}}
    for split in ("train", "validation", "test"):
        if split not in dataset:
            raise ValueError(f"Missing official split: {split}")
        frame = dataset[split].to_pandas().rename(
            columns={"sentence": "text", "sentiment": "label"}
        )[["text", "label"]]
        if frame.isna().any().any() or not frame["text"].map(
            lambda text: isinstance(text, str) and bool(text.strip())
        ).all():
            raise ValueError(f"{split}: missing/blank data; refusing to impute original data")
        if set(frame["label"]) != {0, 1, 2}:
            raise ValueError(f"{split}: unexpected sentiment labels")
        contents = frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
        path = output_dir / f"{split}.csv"
        if path.exists():
            if path.read_bytes() != contents:
                raise FileExistsError(f"{path} differs from source; refusing to overwrite")
            print(f"Keeping identical existing file: {path}")
        else:
            path.write_bytes(contents)
        counts = frame["label"].value_counts().sort_index().to_dict()
        manifest["splits"][split] = {
            "rows": len(frame), "columns": list(frame.columns),
            "label_counts": {str(k): int(v) for k, v in counts.items()},
            "sha256": hashlib.sha256(contents).hexdigest(),
            "source_fingerprint": dataset[split]._fingerprint,
        }
        print(f"{split}: {len(frame)} rows, columns={list(frame.columns)}, labels={counts}", flush=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    download_and_save_data()
