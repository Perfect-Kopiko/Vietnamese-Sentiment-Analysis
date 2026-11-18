"""
Module tiền xử lý và làm sạch văn bản tiếng Việt.
Bao gồm chuẩn hóa Unicode NFC, chuyển chữ thường, loại bỏ thẻ HTML, URL,
rút gọn ký tự lặp kéo dài và chuẩn hóa khoảng trắng.
"""

import re
import unicodedata
from typing import Any


# Biểu thức chính quy cho URL và HTML tags
URL_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE
)
HTML_PATTERN = re.compile(r"<[^>]+>")

# Ký tự lặp lại liên tiếp từ 2 lần trở lên (ví dụ: hayyyyy -> hay, chánnnn -> chán)
CHAR_REPEAT_PATTERN = re.compile(r"(.)\1{1,}", re.IGNORECASE)

# Chuẩn hóa khoảng trắng
WHITESPACE_PATTERN = re.compile(r"\s+")

# Bảng các nguyên âm có dấu tương ứng với nguyên âm gốc trong tiếng Việt
_ACCENTED_VOWELS = {
    "a": "áàảãạăắằẳẵặâấầẩẫậ",
    "e": "éèẻẽẹêếềểễệ",
    "i": "íìỉĩị",
    "o": "óòỏõọôốồổỗộơớờởỡợ",
    "u": "úùủũụưứừửữự",
    "y": "ýỳỷỹỵ"
}


def reduce_elongated_chars(text: str) -> str:
    """
    Rút gọn các ký tự kéo dài ở cuối từ hoặc trong từ:
    - Ký tự giống nhau lặp nhiều lần: 'hayyyyy' -> 'hay', 'chánnnn' -> 'chán'
    - Ký tự nguyên âm có dấu kèm nguyên âm không dấu kéo dài: 'quáaaaa' -> 'quá', 'trờiii' -> 'trời'
    """
    # 1. Rút gọn ký tự trùng lặp liên tiếp: (.)\1+ thành 1 ký tự
    text = CHAR_REPEAT_PATTERN.sub(r"\1", text)

    # 2. Xử lý trường hợp gõ kéo dài nguyên âm có dấu với không dấu (áaaaa -> á, êeeee -> ê)
    for base, variants in _ACCENTED_VOWELS.items():
        for variant in variants:
            # áaaaa -> á
            text = re.sub(rf"({variant}){base}+", r"\1", text, flags=re.IGNORECASE)
            # aaaaaá -> á
            text = re.sub(rf"{base}+({variant})", r"\1", text, flags=re.IGNORECASE)

    return text


def clean_text(text: Any, keep_punctuation: bool = True) -> str:
    """
    Làm sạch văn bản đầu vào:
    1. Chuẩn hóa về chuỗi, kiểm tra rỗng / null.
    2. Chuẩn hóa bảng mã Unicode về dạng NFC (chuẩn hiển thị tiếng Việt dựng sẵn).
    3. Chuyển về chữ thường.
    4. Xóa các URL và thẻ HTML.
    5. Rút gọn các ký tự lặp kéo dài (ví dụ: 'hayyyyy' -> 'hay', 'quáaaaa' -> 'quá', 'chánnnn' -> 'chán').
    6. Xóa hoặc giữ dấu câu tùy chọn, chuẩn hóa khoảng trắng thừa.

    Args:
        text: Chuỗi văn bản cần làm sạch (chấp nhận cả None hoặc kiểu khác).
        keep_punctuation: Có giữ lại dấu câu hay không (mặc định True).

    Returns:
        Chuỗi văn bản đã được làm sạch.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # 1. Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Xóa URL và thẻ HTML
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)

    # 3. Chuyển chữ thường
    text = text.lower()

    # 4. Rút gọn ký tự lặp kéo dài
    text = reduce_elongated_chars(text)

    # 5. Nếu không giữ dấu câu (tùy chọn)
    if not keep_punctuation:
        text = re.sub(r"[^\w\s]", " ", text)

    # 6. Chuẩn hóa khoảng trắng thừa
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


if __name__ == "__main__":
    print("=== TEST MODULE text_cleaner.py ===")
    sample_texts = [
        "Thầy dạy <p>nhiệt tình</p> quáaaaa, giảng bài siêu hayyyyyy <3 http://example.com/slide",
        "Môn này chánnnnn vãiiii, slide dài lê thê     và ko hiểu gì hết!!!",
        "GV giao btvn nhiều kinh khủngggg... xem tại www.fit.uit.edu.vn nhé mn",
        None,
        "   "
    ]

    for idx, sample in enumerate(sample_texts, 1):
        cleaned = clean_text(sample)
        print(f"\n[Mẫu {idx}]")
        print(f"Gốc    : {repr(sample)}")
        print(f"Làm sạch: {repr(cleaned)}")
