"""
Module tách từ tiếng Việt (Vietnamese Word Segmentation).
Bọc thư viện pyvi (ViTokenizer.tokenize) để nối các từ ghép bằng ký tự gạch dưới '_'.
Có cờ 'enabled' để bật/tắt phục vụ nghiên cứu Ablation Study.
"""

from typing import Any


class VietnameseWordSegmenter:
    """
    Class thực hiện tách từ tiếng Việt sử dụng pyvi.ViTokenizer.
    Nếu enabled=True: Tách từ ghép (ví dụ 'giảng viên' -> 'giảng_viên').
    Nếu enabled=False: Giữ nguyên văn bản gốc, dùng cho thí nghiệm so sánh Ablation.
    """

    def __init__(self, enabled: bool = True):
        """
        Khởi tạo VietnameseWordSegmenter.

        Args:
            enabled: Nếu True thì kích hoạt tách từ ghép, ngược lại giữ nguyên.
        """
        self.enabled = enabled
        self._tokenizer = None

        if self.enabled:
            try:
                from pyvi import ViTokenizer
                self._tokenizer = ViTokenizer
            except ImportError:
                print(
                    "[!] Cảnh báo: Thư viện 'pyvi' chưa được cài đặt. "
                    "Hãy chạy 'pip install pyvi'. Sẽ giữ nguyên văn bản mà không tách từ ghép."
                )
                self._tokenizer = None

    def segment(self, text: Any) -> str:
        """
        Tách từ ghép tiếng Việt cho chuỗi đầu vào.

        Args:
            text: Chuỗi văn bản tiếng Việt.

        Returns:
            Chuỗi văn bản đã được phân tách từ ghép nếu enabled=True.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)

        if not text.strip() or not self.enabled or self._tokenizer is None:
            return text

        try:
            return self._tokenizer.tokenize(text)
        except Exception as exc:
            print(f"[!] Lỗi trong quá trình phân tách từ bằng pyvi: {exc}")
            return text

    def __call__(self, text: Any) -> str:
        return self.segment(text)


if __name__ == "__main__":
    print("=== TEST MODULE word_segmenter.py ===")
    segmenter_enabled = VietnameseWordSegmenter(enabled=True)
    segmenter_disabled = VietnameseWordSegmenter(enabled=False)

    sample_sentences = [
        "giảng viên nhiệt tình, sinh viên tiếp thu bài học rất tốt trên lớp.",
        "trường đại học công nghệ thông tin có cơ sở vật chất hiện đại.",
        "bài tập lớn môn học sâu đòi hỏi nhiều kỹ năng lập trình mô hình."
    ]

    for idx, sentence in enumerate(sample_sentences, 1):
        seg_on = segmenter_enabled(sentence)
        seg_off = segmenter_disabled(sentence)
        print(f"\n[Câu {idx}]")
        print(f"Gốc             : {sentence}")
        print(f"Tách từ (ON)    : {seg_on}")
        print(f"Không tách (OFF): {seg_off}")
