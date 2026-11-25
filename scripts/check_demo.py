"""Đối chiếu kết quả demo với checkpoint."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import pandas as pd
from gradio_client import Client

from demo.app import EXAMPLES
from demo.inference_pipeline import InferencePipeline, MAX_TEXT_LENGTH, MODEL_NAMES
from src.models.tfidf_baseline import LABELS, ROOT
from src.datasets.tfidf_data import sha256_file


def check(url="http://127.0.0.1:7860", output_dir=ROOT / ".cache/demo_check"):
    with urlopen(url, timeout=30) as response:
        assert response.status == 200
    client = Client(url, verbose=False, download_files=False, max_workers=2, analytics_enabled=False)
    inference = InferencePipeline()
    if inference.artifact is None:
        raise RuntimeError(inference.status(MODEL_NAMES[0]))
    rows, timings = [], []
    if inference.bilstm_checkpoint.is_file() and inference.bilstm is None:
        raise RuntimeError(inference.status("BiLSTM"))
    for intended, text in EXAMPLES.items():
        for normalize in (False, True):
            start = perf_counter()
            output = client.predict(text, normalize, api_name="/compare")
            timings.append((perf_counter() - start) * 1000)
            assert len(output) == 6
            for i, expected in enumerate(inference.compare(text, normalize)):
                label_output, status = output[i * 2:i * 2 + 2]
                assert status == expected["status"]
                if expected["probabilities"] is None:
                    assert label_output is None or label_output == {}
                    assert "Chưa sẵn sàng" in status
                    continue
                assert label_output["label"] == expected["label"]
                probabilities = {entry["label"]: entry["confidence"]
                                 for entry in label_output["confidences"]}
                assert set(probabilities) == set(LABELS.values())
                for label in LABELS.values():
                    np.testing.assert_allclose(probabilities[label], expected["probabilities"][label],
                                               rtol=0, atol=1e-6 if i == 1 else 1e-12)
                np.testing.assert_allclose(sum(probabilities.values()), 1.0, rtol=0, atol=1e-6)
                rows.append({"intended_group": intended, "text": text, "model": expected["model"],
                             "normalize_noise": normalize, "prediction": expected["label"],
                             **{f"probability_{j}": probabilities[LABELS[j]] for j in [0, 1, 2]}})
    for invalid, message in (("   ", "Vui lòng nhập"), ("a" * (MAX_TEXT_LENGTH + 1), "ký tự")):
        try:
            client.predict(invalid, False, api_name="/compare")
        except Exception as exc:
            if message not in str(exc):
                raise
        else:
            raise AssertionError(f"Invalid input of length {len(invalid)} was accepted")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "demo_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "demo_verification.json").write_text(json.dumps({
        "url": url, "http_status": 200, "n_examples": len(EXAMPLES), "n_requests": len(timings),
        "n_predictions": len(rows), "normalization_modes": [False, True],
        "api_matches_checkpoint": True, "probabilities_sum_to_one": True,
        "blank_input_rejected": True, "long_input_rejected": True,
        "bilstm_ready": inference.bilstm is not None, "phobert_ready": False,
        "checkpoint": str(inference.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(inference.checkpoint),
        "bilstm_checkpoint": inference.bilstm_checkpoint.name,
        "bilstm_checkpoint_sha256": (sha256_file(inference.bilstm_checkpoint)
                                      if inference.bilstm_checkpoint.is_file() else None),
        "api_roundtrip_ms": timings,
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("HTTP, TF-IDF/BiLSTM, normalization toggle, missing PhoBERT and invalid input: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/demo_check")
    args = parser.parse_args()
    check(args.url, args.output_dir)
