"""
Module tạo dữ liệu nhiễu nhân tạo (Noise Simulator) để phục vụ các thí nghiệm mở rộng,
kiểm tra tính chống chịu (robustness) của các mô hình phân tích cảm xúc:
1. Xóa dấu tiếng Việt (remove_accents).
2. Chèn nhiễu teencode ngược từ văn bản chuẩn (inject_teencode).
"""

import json
import os
import random
import re
import unicodedata
from typing import Any, Dict, List, Optional


# Bản đồ chuyển ký tự có dấu sang không dấu
_VIETNAMESE_ACCENT_MAP = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ă": "a", "ằ": "a", "ắ": "a", "ẳ": "a", "ẵ": "a", "ặ": "a",
    "â": "a", "ầ": "a", "ấ": "a", "ẩ": "a", "ẫ": "a", "ậ": "a",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ê": "e", "ề": "e", "ế": "e", "ể": "e", "ễ": "e", "ệ": "e",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ô": "o", "ồ": "o", "ố": "o", "ổ": "o", "ỗ": "o", "ộ": "o",
    "ơ": "o", "ờ": "o", "ớ": "o", "ở": "o", "ỡ": "o", "ợ": "o",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ư": "u", "ừ": "u", "ứ": "u", "ử": "u", "ữ": "u", "ự": "u",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
    "đ": "d",
    "À": "A", "Á": "A", "Ả": "A", "Ã": "A", "Ạ": "A",
    "Ă": "A", "Ằ": "A", "Ắ": "A", "Ẳ": "A", "Ẵ": "A", "Ặ": "A",
    "Â": "A", "Ầ": "A", "Ấ": "A", "Ẩ": "A", "Ẫ": "A", "Ậ": "A",
    "È": "E", "É": "E", "Ẻ": "E", "Ẽ": "E", "Ẹ": "E",
    "Ê": "E", "Ề": "E", "Ế": "E", "Ể": "E", "Ễ": "E", "Ệ": "E",
    "Ì": "I", "Í": "I", "Ỉ": "I", "Ĩ": "I", "Ị": "I",
    "Ò": "O", "Ó": "O", "Ỏ": "O", "Õ": "O", "Ọ": "O",
    "Ô": "O", "Ồ": "O", "Ố": "O", "Ổ": "O", "Ỗ": "O", "Ộ": "O",
    "Ơ": "O", "Ờ": "O", "Ớ": "O", "Ở": "O", "Ỡ": "O", "Ợ": "O",
    "Ù": "U", "Ú": "U", "Ủ": "U", "Ũ": "U", "Ụ": "U",
    "Ư": "U", "Ừ": "U", "Ứ": "U", "Ử": "U", "Ữ": "U", "Ự": "U",
    "Ỳ": "Y", "Ý": "Y", "Ỷ": "Y", "Ỹ": "Y", "Ỵ": "Y",
    "Đ": "D"
}


def remove_accents(text: Any) -> str:
    """
    Xóa toàn bộ dấu tiếng Việt từ văn bản đầu vào.
    Ví dụ: 'thầy giảng dễ hiểu' -> 'thay giang de hieu'

    Args:
        text: Chuỗi văn bản tiếng Việt có dấu.

    Returns:
        Chuỗi văn bản không dấu.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Chuẩn hóa về NFC trước khi map
    normalized = unicodedata.normalize("NFC", text)
    result = "".join(_VIETNAMESE_ACCENT_MAP.get(char, char) for char in normalized)
    return result


def _build_reverse_teencode_map(dict_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Tạo ánh xạ đảo ngược: từ chuẩn -> danh sách các từ teencode tương ứng.
    Ví dụ: 'không' -> ['k', 'ko', 'kh', 'hok']
    """
    if dict_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dict_path = os.path.abspath(
            os.path.join(current_dir, "..", "..", "data", "resources", "teencode_dict.json")
        )

    reverse_map: Dict[str, List[str]] = {}
    if not os.path.exists(dict_path):
        return reverse_map

    try:
        with open(dict_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for teencode, standard in data.items():
            standard_norm = standard.strip().lower()
            teencode_norm = teencode.strip().lower()
            if standard_norm not in reverse_map:
                reverse_map[standard_norm] = []
            if teencode_norm not in reverse_map[standard_norm]:
                reverse_map[standard_norm].append(teencode_norm)
    except Exception as exc:
        print(f"[!] Lỗi khi nạp từ điển đảo ngược teencode: {exc}")

    return reverse_map


# Cache reverse map
_CACHED_REVERSE_MAP: Optional[Dict[str, List[str]]] = None


def inject_teencode(
    text: Any,
    prob: float = 0.3,
    dict_path: Optional[str] = None,
    seed: Optional[int] = None
) -> str:
    """
    Thay thế ngược ngẫu nhiên một phần từ chuẩn thành teencode theo xác suất `prob`.

    Args:
        text: Chuỗi văn bản chuẩn.
        prob: Xác suất thay thế từ (mặc định 0.3 = 30%).
        dict_path: Đường dẫn tệp teencode_dict.json.
        seed: Random seed cho việc tái lập kết quả thí nghiệm.

    Returns:
        Chuỗi văn bản đã được chèn nhiễu teencode.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    global _CACHED_REVERSE_MAP
    if _CACHED_REVERSE_MAP is None or dict_path is not None:
        rev_map = _build_reverse_teencode_map(dict_path)
        if dict_path is None:
            _CACHED_REVERSE_MAP = rev_map
    else:
        rev_map = _CACHED_REVERSE_MAP

    if not rev_map or prob <= 0:
        return text

    rng = random.Random(seed)

    # Tách từ theo khoảng trắng, bảo toàn dấu câu xung quanh
    words = text.split()
    simulated_words = []

    for word in words:
        # Tách phần dấu câu đầu/cuối từ nếu có
        match = re.match(r"^([^\w\s]*)([\w-]+)([^\w\s]*)$", word, re.UNICODE)
        if match:
            prefix, core_word, suffix = match.groups()
            core_lower = core_word.lower()
            if core_lower in rev_map and rng.random() < prob:
                chosen_teencode = rng.choice(rev_map[core_lower])
                # Bảo toàn viết hoa nếu từ gốc viết hoa toàn bộ hoặc chữ đầu
                if core_word.isupper():
                    chosen_teencode = chosen_teencode.upper()
                elif core_word.istitle():
                    chosen_teencode = chosen_teencode.capitalize()
                simulated_words.append(f"{prefix}{chosen_teencode}{suffix}")
            else:
                simulated_words.append(word)
        else:
            simulated_words.append(word)

    return " ".join(simulated_words)


if __name__ == "__main__":
    print("=== TEST MODULE noise_simulator.py ===")
    sample_text = "Thầy giảng viên dạy bài tập rất hay nhưng kiểm tra hơi khó, sinh viên không hiểu gì."

    print(f"Văn bản gốc: {sample_text}\n")

    # 1. Test remove_accents
    no_accent = remove_accents(sample_text)
    print(f"[-] remove_accents:\n    {no_accent}\n")

    # 2. Test inject_teencode với các mức xác suất khác nhau
    for p in [0.3, 0.6, 1.0]:
        noisy_text = inject_teencode(sample_text, prob=p, seed=42)
        print(f"[-] inject_teencode (prob={p}):\n    {noisy_text}")
