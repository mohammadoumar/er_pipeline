import os
import gradio as gr
from PIL import Image

from models.llama_vision import LlamaVisionModel
from models.qwen_vl import QwenVLModel
from inference.pipeline import EmotionPipeline

# --- Model paths (replace with actual paths before running) ---
MODEL_PATHS = {
    "LLaMA-3.2-Vision": "/path/to/llama-3.2-vision-finetuned",
    "Qwen2.5-VL":        "/path/to/qwen2.5-vl-finetuned",
}

MODEL_CLASSES = {
    "LLaMA-3.2-Vision": LlamaVisionModel,
    "Qwen2.5-VL":        QwenVLModel,
}

loaded_models: dict = {}


def get_model(model_name: str):
    if model_name not in loaded_models:
        cls = MODEL_CLASSES[model_name]
        path = MODEL_PATHS[model_name]
        loaded_models[model_name] = cls(path)
    return loaded_models[model_name]


def process(image: Image.Image, model_name: str, page_name: str):
    if image is None:
        return "Please upload a comics page.", "", ""

    page_name = page_name.strip() or "page"
    model = get_model(model_name)
    pipeline = EmotionPipeline(model)
    results = pipeline.run(image, page_name=page_name)

    # Panel results
    panel_lines = []
    for p in results["panels"]:
        panel_lines.append(
            f"**Panel {p['panel_id'] + 1}**\n"
            f"- Utterance: _{p['utterance'] or '(none)'}_ \n"
            f"- Emotion: `{p['emotion']}`\n"
        )
    panel_text = "\n---\n".join(panel_lines)

    # Dataset info
    dataset_info = (
        f"Dataset saved to: `{results['dataset_path']}`\n"
        f"Panels: {results['num_panels']}\n"
        f"Columns: page_name, panel_id, image, utterance"
    )

    return panel_text, results["narrative"], dataset_info


with gr.Blocks(title="Comics Emotion Recognition") as demo:
    gr.Markdown(
        "# Comics Emotion Recognition\n"
        "Upload a comics page. The system will detect panels, extract utterances via OCR, "
        "build a dataset, and generate per-panel emotion labels and a full-page narrative."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Comics Page")
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CLASSES.keys()),
                value="Qwen2.5-VL",
                label="Select Vision Model",
            )
            page_name_input = gr.Textbox(
                value="page_01",
                label="Page Name (used for dataset save path)",
            )
            run_btn = gr.Button("Analyze", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("Panel Emotions"):
                    panel_output = gr.Markdown()
                with gr.Tab("Page Narrative"):
                    narrative_output = gr.Textbox(lines=10, label="Emotion Narrative")
                with gr.Tab("Dataset Info"):
                    dataset_output = gr.Textbox(lines=5, label="Dataset Info")

    run_btn.click(
        fn=process,
        inputs=[image_input, model_dropdown, page_name_input],
        outputs=[panel_output, narrative_output, dataset_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7861)),
        share=True,
    )
