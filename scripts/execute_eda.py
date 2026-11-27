"""Chạy notebook EDA với một kernel mới."""

import argparse
import os
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

for key, folder in (("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter"),
                    ("MPLCONFIGDIR", "matplotlib")):
    os.environ.setdefault(key, str(ROOT / ".cache" / folder))


def execute(repeats=2):
    path = ROOT / "notebooks/01_eda_data_stats.ipynb"
    for run in range(1, repeats + 1):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        manager = KernelManager(kernel_name="python3")
        # Dùng Python hiện tại thay cho kernel mặc định của máy.
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, km=manager, timeout=300, allow_errors=False,
                                resources={"metadata": {"path": str(ROOT)}})
        with client.setup_kernel(cleanup_kc=True):
            client.execute()
        nbformat.write(notebook, path)
        count = sum(cell.cell_type == "code" for cell in notebook.cells)
        print(f"EDA execution {run}/{repeats}: {count} code cells OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    execute(args.repeats)
