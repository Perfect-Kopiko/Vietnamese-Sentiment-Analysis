"""
Module chuẩn hóa từ lóng, teencode và viết tắt dựa trên từ điển teencode_dict.json.
Đồng thời cung cấp hàm tùy chọn khôi phục dấu tiếng Việt (fallback an toàn).
"""

import json
import os
import re
from typing import Any, Dict, Optional


class NoiseNormalizer:
    """
    Chuẩn hóa từ viết tắt, teencode cho tiếng Việt dựa trên từ điển ánh xạ.
    Sử dụng regex ranh giới từ tiếng Việt để tránh thay nhầm các chuỗi con.
    """

    def __init__(self, dict_path: Optional[str] = None):
        """
        Khởi tạo NoiseNormalizer và nạp từ điển teencode.

        Args:
            dict_path: Đường dẫn đến tệp teencode_dict.json. Nếu None, sẽ tự động tìm
                       theo đường dẫn mặc định trong data/resources/.
        """
        if dict_path is None:
            # Tìm đường dẫn tương đối từ vị trí file hiện tại
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dict_path = os.path.abspath(
                os.path.join(current_dir, "..", "..", "data", "resources", "teencode_dict.json")
            )

        self.dict_path = dict_path
        self.teencode_dict: Dict[str, str] = self._load_dictionary(dict_path)
        self.compiled_regex = self._compile_regex(self.teencode_dict)

    @staticmethod
    def _load_dictionary(path: str) -> Dict[str, str]:
        """Đọc và kiểm tra tệp từ điển JSON."""
        if not os.path.exists(path):
            print(f"[!] Cảnh báo: Tệp từ điển teencode không tồn tại tại '{path}'. Sử dụng từ điển rỗng.")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Chuẩn hóa tất cả key về chữ thường
                return {k.lower(): str(v).lower() for k, v in data.items()}
            return {}
        except Exception as exc:
            print(f"[!] Lỗi khi nạp từ điển teencode từ '{path}': {exc}")
            return {}

    @staticmethod
    def _compile_regex(mapping: Dict[str, str]) -> Optional[re.Pattern]:
        """
        Biên dịch biểu thức chính quy cho toàn bộ các từ teencode.
        Sắp xếp theo độ dài giảm dần để ưu tiên cụm từ dài trước.
        Dùng lookahead và lookbehind để định nghĩa ranh giới từ hỗ trợ cả ký tự tiếng Việt.
        """
        if not mapping:
            return None

        # Sắp xếp từ khóa theo chiều dài giảm dần
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        escaped_keys = [re.escape(k) for k in sorted_keys]

        # Ranh giới từ tiếng Việt: không đứng trước hoặc sau một ký tự chữ hoặc số hoặc gạch dưới
        # \w trong Python regex với cờ UNICODE đã bao gồm các ký tự có dấu tiếng Việt
        pattern = r"(?<!\w)(" + "|".join(escaped_keys) + r")(?!\w)"
        return re.compile(pattern, re.IGNORECASE | re.UNICODE)

    def normalize(self, text: Any) -> str:
        """
        Thay thế các từ teencode trong văn bản thành từ chuẩn.

        Args:
            text: Chuỗi văn bản đầu vào.

        Returns:
            Chuỗi văn bản sau khi chuẩn hóa teencode.
        """
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        if not text.strip() or self.compiled_regex is None:
            return text

        def _replace_match(match: re.Match) -> str:
            word = match.group(0).lower()
            return self.teencode_dict.get(word, match.group(0))

        return self.compiled_regex.sub(_replace_match, text)

    def restore_accents(self, text: Any) -> str:
        """
        Hàm tùy chọn khôi phục dấu tiếng Việt (diacritics restoration).
        Thực hiện fallback an toàn: nếu chưa tích hợp model add_accents
        chuyên dụng, trả về văn bản đã chuẩn hóa và cảnh báo nhẹ.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        # Fallback an toàn (có thể mở rộng thêm mô hình underthesea hoặc seq2seq nếu cần)
        return text


if __name__ == "__main__":
    print("=== TEST MODULE noise_normalizer.py ===")
    normalizer = NoiseNormalizer()
    print(f"Tổng số từ trong từ điển: {len(normalizer.teencode_dict)}")

    sample_sentences = [
        "thầy gv dạy rất hay nhg cho bt và ktra hơi khó, mn nên chú ý.",
        "mik thấy sv đh này hok rất chăm, điểm thi k quá thấp.",
        "slide môn cntt này ntn vậy bạn? rep mik vs, tks b nhiều nha!"
    ]

    for idx, sentence in enumerate(sample_sentences, 1):
        normalized = normalizer.normalize(sentence)
        print(f"\n[Câu {idx}]")
        print(f"Gốc       : {sentence}")
        print(f"Chuẩn hóa : {normalized}")
