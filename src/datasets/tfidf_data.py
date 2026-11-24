"""Đọc và kiểm tra dữ liệu gốc."""

import hashlib
from pathlib import Path

import pandas as pd


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(path):
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    if list(frame.columns) != ["text", "label"]:
        raise ValueError(f"{path}: expected columns text,label; got {list(frame.columns)}")
    if frame.empty or not frame["text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError(f"{path}: empty split or missing/blank text")
    if not pd.api.types.is_integer_dtype(frame["label"]) or set(frame["label"]) != {0, 1, 2}:
        raise ValueError(f"{path}: labels must be integers 0,1,2, with all classes present")
    return frame
