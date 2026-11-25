"""Huấn luyện baseline, chọn trên validation và đánh giá trên test."""

import argparse
import itertools
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from src.datasets.tfidf_data import read_split, sha256_file
from src.evaluation.metrics import classification_details, classification_metrics
from src.models.tfidf_baseline import ROOT, build_pipeline
from src.preprocessing.noise_simulator import inject_teencode, remove_accents


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_comparison(results, output_dir):
    table = results.loc[results["split"] == "test"].pivot(
        index=["experiment", "use_segment", "normalize_noise", "fold_accents", "C", "selected_for_demo"],
        columns="scenario", values=["accuracy", "macro_f1"],
    )
    table.columns = [f"{metric}_{scenario}" for metric, scenario in table.columns]
    table.reset_index().to_csv(output_dir / "test_comparison.csv", index=False)


def make_scenarios(texts, seed, probability):
    """Tạo cùng các bản nhiễu cho mọi cấu hình."""
    return {
        "clean": list(texts),
        "no_accents": [remove_accents(text) for text in texts],
        "teencode": [inject_teencode(text, prob=probability, seed=seed + i)
                     for i, text in enumerate(texts)],
    }


def train(config_path=ROOT / "configs/tfidf_lr.yaml", *, output_dir=None, checkpoint=None):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if output_dir is not None:
        config["output_dir"] = str(output_dir)
    if checkpoint is not None:
        config["checkpoint"] = str(checkpoint)
    data_dir, output_dir = ROOT / config["data_dir"], ROOT / config["output_dir"]
    checkpoint = ROOT / config["checkpoint"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # Ghi checksum trước khi chạy. Chỉ nạp tập test sau khi lưu checkpoint.
    hashes = {name: sha256_file(data_dir / f"{name}.csv")
              for name in ("train", "validation", "test")}
    train_frame = read_split(data_dir / "train.csv")
    validation = read_split(data_dir / "validation.csv")
    search_rows, selected_models = [], []
    flags = ("use_segment", "normalize_noise", "fold_accents")
    combinations = itertools.product(*(config["experiments"][flag] for flag in flags))
    for combination in combinations:
        options = dict(zip(flags, combination))
        experiment = "_".join(f"{key}{int(value)}" for key, value in options.items())
        best = None
        for C in config["logistic_regression"]["C_values"]:
            pipeline = build_pipeline(config, **options, C=C)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(train_frame["text"].tolist(), train_frame["label"])
            prediction = pipeline.predict(validation["text"].tolist())
            metrics = classification_metrics(validation["label"], prediction)
            row = {"experiment": experiment, **options, "C": C, **metrics}
            search_rows.append(row)
            print(f"validation {experiment} C={C}: {metrics}", flush=True)
            # Giữ C xuất hiện trước nếu Macro-F1 bằng nhau.
            if best is None or metrics["macro_f1"] > best[0]["macro_f1"]:
                best = (row, pipeline)
        selected_models.append(best)

    best_row, best_pipeline = max(selected_models, key=lambda item: item[0]["macro_f1"])
    artifact = {
        "format_version": 1, "pipeline": best_pipeline,
        "selection": best_row, "selection_split": "validation",
        "selection_metric": "macro_f1", "config": config, "data_sha256": hashes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    pd.DataFrame(search_rows).to_csv(output_dir / "validation_search.csv", index=False)
    write_json(output_dir / "selection.json", {key: value for key, value in artifact.items()
                                              if key != "pipeline"})
    joblib.dump(artifact, checkpoint, compress=3)
    print(f"Selected using validation ONLY: {best_row}", flush=True)

    test = read_split(data_dir / "test.csv")
    result_rows, details = [], {}
    for split, frame in (("validation", validation), ("test", test)):
        scenarios = make_scenarios(frame["text"].tolist(), config["seed"],
                                   config["noise"]["teencode_probability"])
        for row, pipeline in selected_models:
            for scenario, texts in scenarios.items():
                predictions = pipeline.predict(texts)
                changed = sum(a != b for a, b in zip(frame["text"], texts))
                result_rows.append({
                    "experiment": row["experiment"], **{key: row[key] for key in flags},
                    "C": row["C"], "selected_for_demo": row["experiment"] == best_row["experiment"],
                    "split": split, "scenario": scenario, "n_samples": len(frame),
                    "changed_samples": changed, "changed_fraction": changed / len(frame),
                    **classification_metrics(frame["label"], predictions),
                })
                details[f"{row['experiment']}/{split}/{scenario}"] = classification_details(
                    frame["label"], predictions
                )
                if row["experiment"] == best_row["experiment"] and split == "test":
                    probabilities = pipeline.predict_proba(texts)
                    pd.DataFrame({
                        "row_id": np.arange(len(frame)), "label": frame["label"],
                        "prediction": predictions,
                        **{f"probability_{int(label)}": probabilities[:, i]
                           for i, label in enumerate(pipeline.classes_)},
                    }).to_csv(output_dir / f"predictions_test_{scenario}.csv", index=False)

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "experiment_results.csv", index=False)
    write_comparison(results, output_dir)
    write_json(output_dir / "classification_reports.json", details)
    versions = {name: version(name) for name in (
        "scikit-learn", "numpy", "pandas", "pyvi", "joblib", "PyYAML",
        "datasets", "gradio", "matplotlib", "seaborn", "nbclient", "ipykernel",
    )}
    write_json(output_dir / "run_metadata.json", {
        "python": platform.python_version(), "packages": versions, "data_sha256": hashes,
        "dictionary_sha256": sha256_file(ROOT / "data/resources/teencode_dict.json"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "noise_seed": config["seed"], "noise_probability": config["noise"]["teencode_probability"],
        "protocol": "Fit train only; select C and demo model on clean validation macro-F1; "
                    "freeze selection before test; no refit on validation; noise only at evaluation.",
        "normalization": "Teencode dictionary; optional accent folding after optional PyVi. "
                         "No accent restoration. Normalization variants are separately trained.",
        "teencode_scope": "Existing simulator replaces eligible single tokens, not multiword phrases.",
    })
    for name, before in hashes.items():
        if sha256_file(data_dir / f"{name}.csv") != before:
            raise RuntimeError(f"Original data changed during training: {name}")
    print(results.loc[results["split"] == "test"].to_string(index=False), flush=True)
    print(f"Saved checkpoint: {checkpoint}\nResults: {output_dir}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tfidf_lr.yaml")
    parser.add_argument("--output-dir", type=Path, help="Write experiment reports to a separate directory.")
    parser.add_argument("--checkpoint", type=Path, help="Save the selected model at a separate path.")
    args = parser.parse_args()
    if (args.output_dir is None) != (args.checkpoint is None):
        parser.error("Pass --output-dir and --checkpoint together to use separate paths.")
    train(args.config, output_dir=args.output_dir, checkpoint=args.checkpoint)
