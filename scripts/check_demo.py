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
    for intended, text in EXAMPLES.items():
        start = perf_counter()
        output = client.predict(text, api_name="/compare")
        timings.append((perf_counter() - start) * 1000)
        expected = inference.predict(text)
        assert len(output) == 6
        assert output[0]["label"] == expected["label"]
        probabilities = {entry["label"]: entry["confidence"] for entry in output[0]["confidences"]}
        assert set(probabilities) == set(LABELS.values())
        for label in LABELS.values():
            np.testing.assert_allclose(probabilities[label], expected["probabilities"][label], atol=1e-12)
        np.testing.assert_allclose(sum(probabilities.values()), 1.0, atol=1e-12)
        for probability_index, status_index in ((2, 3), (4, 5)):
            # Gradio có thể trả {} hoặc None khi chưa có kết quả.
            empty_label = output[probability_index]
            assert empty_label is None or empty_label == {}
            assert "Chưa sẵn sàng" in output[status_index]
        rows.append({"intended_group": intended, "text": text, "prediction": expected["label"],
                     **{f"probability_{i}": probabilities[LABELS[i]] for i in [0, 1, 2]}})
    for invalid, message in (("   ", "Vui lòng nhập"), ("a" * (MAX_TEXT_LENGTH + 1), "ký tự")):
        try:
            client.predict(invalid, api_name="/compare")
        except Exception as exc:
            if message not in str(exc):
                raise
        else:
            raise AssertionError(f"Invalid input of length {len(invalid)} was accepted")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "demo_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "demo_verification.json").write_text(json.dumps({
        "url": url, "http_status": 200, "n_examples": len(rows),
        "api_matches_checkpoint": True, "probabilities_sum_to_one": True,
        "blank_input_rejected": True, "long_input_rejected": True,
        "bilstm_ready": False, "phobert_ready": False,
        "checkpoint": str(inference.checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(inference.checkpoint),
        "api_roundtrip_ms": timings,
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("HTTP, API/checkpoint parity, unavailable models and invalid input: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".cache/demo_check")
    args = parser.parse_args()
    check(args.url, args.output_dir)
