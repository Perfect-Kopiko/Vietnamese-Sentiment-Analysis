"""
Package preprocessing chứa các module tiền xử lý văn bản tiếng Việt:
- clean_text: Làm sạch, chuẩn hóa NFC, loại URL/HTML, rút gọn ký tự lặp.
- NoiseNormalizer: Chuẩn hóa từ lóng, teencode, viết tắt.
- remove_accents, inject_teencode: Giả lập nhiễu cho thí nghiệm mở rộng.
- VietnameseWordSegmenter: Tách từ ghép tiếng Việt bằng pyvi.
- preprocess_pipeline: Pipeline tích hợp toàn bộ các bước tiền xử lý.
"""

import sys
import os
from typing import Any, Optional

# Đảm bảo thư mục gốc dự án có trong sys.path khi chạy trực tiếp file này
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.preprocessing.text_cleaner import clean_text
from src.preprocessing.noise_normalizer import NoiseNormalizer
from src.preprocessing.noise_simulator import remove_accents, inject_teencode
from src.preprocessing.word_segmenter import VietnameseWordSegmenter


# Khởi tạo sẵn các singleton instances phục vụ pipeline mặc định
_DEFAULT_NORMALIZER: Optional[NoiseNormalizer] = None
_DEFAULT_SEGMENTER_ON: Optional[VietnameseWordSegmenter] = None


def preprocess_pipeline(
    text: Any,
    use_segment: bool = True,
    normalize_noise: bool = False
) -> str:
    """
    Pipeline tiền xử lý toàn diện cho văn bản tiếng Việt:
    1. Làm sạch cơ bản: NFC, chữ thường, bỏ HTML/URL, rút gọn lặp ký tự (clean_text).
    2. (Tùy chọn) Chuẩn hóa teencode / viết tắt (normalize_noise=True).
    3. (Tùy chọn) Tách từ ghép tiếng Việt (use_segment=True).

    Args:
        text: Văn bản đầu vào.
        use_segment: Có tách từ ghép tiếng Việt bằng pyvi hay không.
        normalize_noise: Có chuẩn hóa teencode sang từ chuẩn hay không.

    Returns:
        Văn bản đã qua pipeline tiền xử lý.
    """
    global _DEFAULT_NORMALIZER, _DEFAULT_SEGMENTER_ON

    # Bước 1: Làm sạch cơ bản
    cleaned = clean_text(text)

    # Bước 2: Chuẩn hóa teencode nếu được yêu cầu
    if normalize_noise:
        if _DEFAULT_NORMALIZER is None:
            _DEFAULT_NORMALIZER = NoiseNormalizer()
        cleaned = _DEFAULT_NORMALIZER.normalize(cleaned)

    # Bước 3: Tách từ ghép nếu được yêu cầu
    if use_segment:
        if _DEFAULT_SEGMENTER_ON is None:
            _DEFAULT_SEGMENTER_ON = VietnameseWordSegmenter(enabled=True)
        cleaned = _DEFAULT_SEGMENTER_ON.segment(cleaned)

    return cleaned


__all__ = [
    "clean_text",
    "NoiseNormalizer",
    "remove_accents",
    "inject_teencode",
    "VietnameseWordSegmenter",
    "preprocess_pipeline",
]


if __name__ == "__main__":
    print("=== TEST MODULE preprocessing/__init__.py (Pipeline) ===")
    sample_tests = [
        "Thầy gv dạy siu hayyyyyy nhg btvn nhìu waaaa mn ơi! <3",
        "Slide môn học k đc rõ ràng, sv hok rất mệt mỏi...",
        "Bài giảng tuyệt vời, thầy cô nhiệt tình hỗ trợ sinh viên hết mình."
    ]

    for idx, sample in enumerate(sample_tests, 1):
        print(f"\n[Ví dụ {idx}] Gốc: {sample}")
        out_basic = preprocess_pipeline(sample, use_segment=False, normalize_noise=False)
        out_norm = preprocess_pipeline(sample, use_segment=False, normalize_noise=True)
        out_full = preprocess_pipeline(sample, use_segment=True, normalize_noise=True)
        print(f"  -> Chỉ Clean               : {out_basic}")
        print(f"  -> Clean + Normalize Teencode: {out_norm}")
        print(f"  -> Full (+ Word Segmentation): {out_full}")
