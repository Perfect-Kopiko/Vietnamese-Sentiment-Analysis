"""Đánh giá checkpoint có sẵn."""

import argparse
from pathlib import Path

import pandas as pd

from scripts.train_tfidf import make_scenarios, write_json
from src.datasets.tfidf_data import read_split, sha256_file
from src.evaluation.metrics import classification_details, classification_metrics
from src.models.tfidf_baseline import ROOT, load_checkpoint


def evaluate(checkpoint=ROOT / "checkpoints/best_tfidf.pkl", split="test", output_dir=None):
    artifact = load_checkpoint(checkpoint)
    config = artifact["config"]
    path = ROOT / config["data_dir"] / f"{split}.csv"
    if sha256_file(path) != artifact["data_sha256"][split]:
        raise ValueError(f"{path}: data does not match the trained checkpoint provenance")
    frame = read_split(path)
    scenarios = make_scenarios(frame["text"].tolist(), config["seed"],
                               config["noise"]["teencode_probability"])
    rows, details = [], {}
    for scenario, texts in scenarios.items():
        prediction = artifact["pipeline"].predict(texts)
        rows.append({"split": split, "scenario": scenario, "n_samples": len(frame),
                     **classification_metrics(frame["label"], prediction)})
        details[scenario] = classification_details(frame["label"], prediction)
    output_dir = Path(output_dir) if output_dir is not None else ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_dir / f"checkpoint_evaluation_{split}.csv", index=False)
    write_json(output_dir / f"checkpoint_evaluation_{split}.json", details)
    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/best_tfidf.pkl")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output-dir", type=Path, help="Write a fresh report outside the saved results.")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split, args.output_dir)
