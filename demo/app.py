"""Demo Gradio so sánh ba mô hình."""

import os
from pathlib import Path

from src.models.tfidf_baseline import ROOT

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_TEMP_DIR", str(ROOT / ".cache/gradio"))

import gradio as gr

from demo.inference_pipeline import InferencePipeline, MAX_TEXT_LENGTH, MODEL_NAMES

EXAMPLES = {
    "Tích cực": "Thầy giảng bài rất dễ hiểu, nhiệt tình và tận tâm với sinh viên.",
    "Tiêu cực": "Giảng viên dạy khó hiểu, thiếu nhiệt tình và thường xuyên đi trễ.",
    "Trung tính": "Môn này có ba tín chỉ.",
    "Teencode": "gv dạy ko rõ ràng, sv k hiểu bài",
    "Không dấu": "thay day rat hay nhung bai tap hoi kho",
}
CSS = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")


def theme():
    return gr.themes.Soft(primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
                          font=["Segoe UI", "Arial", "sans-serif"],
                          font_mono=["Consolas", "monospace"])


def model_cards(inference):
    outputs = []
    with gr.Row(elem_id="model-results", equal_height=True):
        for model in MODEL_NAMES:
            with gr.Column(min_width=280, elem_classes="model-card"):
                gr.Markdown(f"### {model}")
                label = gr.Label(label="Nhãn và xác suất", num_top_classes=3,
                                 min_width=0, color="#0f766e")
                status = gr.Markdown(inference.status(model), elem_classes="model-status")
                outputs.extend([label, status])
    return outputs


def add_main_comparison(inference):
    with gr.Column(elem_classes="input-panel"):
        text = gr.Textbox(label="Bình luận tiếng Việt", lines=3,
                          placeholder="Nhập bình luận hoặc chọn câu mẫu.",
                          max_length=MAX_TEXT_LENGTH)
        with gr.Row(elem_id="example-buttons"):
            for label, sentence in EXAMPLES.items():
                gr.Button(label, size="sm", min_width=100).click(
                    lambda value=sentence: value, outputs=text,
                    queue=False, api_visibility="private",
                )
        with gr.Row(elem_id="demo-actions"):
            run = gr.Button("So sánh mô hình", variant="primary", scale=3)
            clear = gr.Button("Xóa", scale=1)
    outputs = model_cards(inference)

    def predict(text):
        try:
            results = inference.compare(text)
        except ValueError as exc:
            raise gr.Error(str(exc), print_exception=False) from exc
        return [value for result in results for value in (result["probabilities"], result["status"])]

    def reset():
        return ("", *(value for model in MODEL_NAMES for value in (None, inference.status(model))))

    run.click(predict, inputs=text, outputs=outputs, api_name="compare",
              concurrency_limit=1)
    clear.click(reset, outputs=[text, *outputs], queue=False, api_visibility="private")


def create_app(inference=None):
    inference = inference or InferencePipeline()
    with gr.Blocks(title="Phân tích cảm xúc tiếng Việt", analytics_enabled=False) as app:
        with gr.Column(elem_id="demo-shell"):
            gr.HTML('<header class="hero"><h1>Phân tích cảm xúc tiếng Việt</h1>'
                    '<p>So sánh kết quả dự đoán của ba mô hình.</p></header>')
            add_main_comparison(inference)
        app.queue(max_size=8, default_concurrency_limit=1)
    return app


if __name__ == "__main__":
    create_app().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=False, theme=theme(), css=CSS, max_threads=4,
        footer_links=[], state_session_capacity=32,
    )
