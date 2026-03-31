import os
import gradio as gr
from PIL import Image

from ocr.extractor import ComicsOCR
from dataset.builder import DatasetBuilder
from models.llama_vision import LlamaVisionModel
from models.qwen_vl import QwenVLModel
from inference.pipeline import EmotionPipeline

# ── Model paths (update before running) ───────────────────────────────────────
MODEL_PATHS = {
    "LLaMA-3.2-Vision": "/path/to/llama-3.2-vision-finetuned",
    "Qwen2.5-VL":        "/path/to/qwen2.5-vl-finetuned",
}
MODEL_CLASSES = {
    "LLaMA-3.2-Vision": LlamaVisionModel,
    "Qwen2.5-VL":        QwenVLModel,
}

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
_ocr: ComicsOCR | None = None
_inference_models: dict = {}


def get_ocr() -> ComicsOCR:
    global _ocr
    if _ocr is None:
        _ocr = ComicsOCR()
    return _ocr


def get_inference_model(name: str):
    if name not in _inference_models:
        _inference_models[name] = MODEL_CLASSES[name](MODEL_PATHS[name])
    return _inference_models[name]


# ── Gallery update on upload ───────────────────────────────────────────────────

# ── Step 1: Extract ────────────────────────────────────────────────────────────

def extract(img1, img2, img3, img4):
    """
    Accepts up to 4 PIL Images from gr.Image components.
    Yields (summary, table, panels_state, status) tuples for live status updates.
    """
    images = [img for img in [img1, img2, img3, img4] if img is not None]
    if not images:
        yield "Please upload at least one comics image.", "", [], "No files uploaded."
        return

    global _ocr
    if _ocr is None:
        yield "", "", [], "Loading DeepSeek-OCR model... (this may take a minute)"
        _ocr = ComicsOCR()
    yield "", "", [], "Model loaded. Starting extraction..."

    all_panels = []
    for i, image in enumerate(images):
        yield "", "", [], f"Extracting page {i + 1} of {len(images)}..."
        panels = _ocr.process_page(image)
        for panel in panels:
            all_panels.append({"page_id": i, **panel})
        yield "", "", [], f"Page {i + 1}: {len(panels)} panel(s) found."

    yield "", "", [], "Building dataset..."
    builder = DatasetBuilder()
    dataset, path = builder.build_and_save(
        [{k: v for k, v in p.items() if k != "page_id"} for p in all_panels],
        page_name="extracted"
    )

    num_pages      = len(images)
    num_panels     = len(all_panels)
    num_utterances = sum(1 for p in all_panels if p["utterance"])
    summary = (
        f"### Extraction Complete\n"
        f"| | |\n|---|---|\n"
        f"| Pages uploaded | {num_pages} |\n"
        f"| Panels detected | {num_panels} |\n"
        f"| Utterances extracted | {num_utterances} |\n"
        f"| Dataset saved to | `{path}` |\n"
    )

    rows = ["| Page | Panel | Utterance |", "|---|---|---|"]
    for p in all_panels:
        utterance = p["utterance"].replace("\n", " ") or "_(none)_"
        rows.append(f"| {p['page_id'] + 1} | {p['panel_id'] + 1} | {utterance} |")
    table = "\n".join(rows)

    yield summary, table, all_panels, "Extraction complete!"


# ── Step 2: Generate ───────────────────────────────────────────────────────────

def generate(panels_state: list, model_name: str) -> tuple[str, str]:
    if not panels_state:
        return "Run extraction first.", ""

    model = get_inference_model(model_name)

    for p in panels_state:
        p["emotion"] = model.predict_panel(p["image"], p["utterance"])

    pages: dict[int, list] = {}
    for p in panels_state:
        pages.setdefault(p["page_id"], []).append(p)

    narratives = []
    for page_id, page_panels in sorted(pages.items()):
        narratives.append(
            f"**Page {page_id + 1}:**\n{model.generate_narrative(page_panels)}"
        )

    rows = ["| Page | Panel | Utterance | Emotion |", "|---|---|---|---|"]
    for p in panels_state:
        utterance = (p["utterance"] or "_(none)_").replace("\n", " ")
        rows.append(
            f"| {p['page_id'] + 1} | {p['panel_id'] + 1} | {utterance} | `{p['emotion']}` |"
        )
    emotion_table = "\n".join(rows)

    return emotion_table, "\n\n---\n\n".join(narratives)


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="Comics Emotion Recognition") as demo:
    panels_state = gr.State([])

    gr.Markdown("# Comics Emotion Recognition")

    # ── Step 1 ──
    gr.Markdown("## Step 1: Extract Panels & Utterances")
    gr.Markdown("Upload up to 4 comics pages. Click **Extract** to run OCR.")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                img1 = gr.Image(type="pil", label="Page 1", height=200)
                img2 = gr.Image(type="pil", label="Page 2", height=200)
            with gr.Row():
                img3 = gr.Image(type="pil", label="Page 3", height=200)
                img4 = gr.Image(type="pil", label="Page 4", height=200)
            extract_btn = gr.Button("Extract", variant="primary")
            status_box = gr.Textbox(label="Progress", interactive=False, lines=4)

        with gr.Column(scale=2):
            extract_summary = gr.Markdown()
            extract_table = gr.Markdown()

    extract_btn.click(
        fn=extract,
        inputs=[img1, img2, img3, img4],
        outputs=[extract_summary, extract_table, panels_state, status_box],
    )

    gr.Markdown("---")

    # ── Step 2 ──
    gr.Markdown("## Step 2: Generate Emotion Labels & Narrative")
    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CLASSES.keys()),
                value="Qwen2.5-VL",
                label="Select Vision Model",
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Panel Emotions"):
                    emotion_output = gr.Markdown()
                with gr.Tab("Page Narrative"):
                    narrative_output = gr.Textbox(lines=12, label="Emotion Narrative")

    generate_btn.click(
        fn=generate,
        inputs=[panels_state, model_dropdown],
        outputs=[emotion_output, narrative_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7861)),
        share=True,
    )
