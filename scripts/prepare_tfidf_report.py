"""Xuất bảng và Confusion Matrix từ kết quả TF-IDF đã lưu."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.tfidf_baseline import ROOT


def prepare(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / "reports/tfidf"
    details = json.loads((source / "classification_reports.json").read_text(encoding="utf-8"))
    selected = json.loads((source / "selection.json").read_text(encoding="utf-8"))["selection"]
    metrics = pd.read_csv(source / "experiment_results.csv")
    rows = []
    for row in metrics.itertuples():
        report = details[f"{row.experiment}/{row.split}/{row.scenario}"]["classification_report"]
        rows.append({
            "experiment": row.experiment, "split": row.split, "scenario": row.scenario,
            "selected_for_demo": row.selected_for_demo, "C": row.C,
            "accuracy": row.accuracy, "macro_f1": row.macro_f1,
            **{f"f1_{name}": report[name]["f1-score"] for name in ("negative", "neutral", "positive")},
        })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "metrics_all.csv", index=False)
    table[table["selected_for_demo"]].to_csv(output_dir / "metrics_selected.csv", index=False)
    names = ["Neg (0)", "Neu (1)", "Pos (2)"]
    for scenario in ("clean", "no_accents", "teencode"):
        matrix = np.asarray(details[f"{selected['experiment']}/test/{scenario}"]["confusion_matrix"])
        pd.DataFrame(matrix, index=names, columns=names).rename_axis("True / Predicted").to_csv(
            output_dir / f"confusion_matrix_{scenario}.csv"
        )
        fig, ax = plt.subplots(figsize=(5.5, 4.6), layout="constrained")
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", fontsize=12,
                    color="white" if value > matrix.max() / 2 else "#183044")
        ax.set(xticks=range(3), yticks=range(3), xticklabels=names, yticklabels=names,
               xlabel="Predicted label", ylabel="True label",
               title=f"TF-IDF + Logistic Regression — test {scenario}")
        fig.savefig(output_dir / f"confusion_matrix_{scenario}.png", dpi=300)
        plt.close(fig)
    print(f"Saved tables and confusion matrices: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/handoff/tfidf")
    prepare(parser.parse_args().output_dir)
