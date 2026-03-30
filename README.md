# Comics Emotion Recognition

A multimodal emotion recognition system for comics pages. Upload a comics page, extract panels and utterances via OCR, and generate per-panel emotion labels and a full-page emotion narrative using fine-tuned vision-language models.

## Pipeline

```
Comics Page Image
       │
       ▼
Panel Detection (Kumiko)
       │
       ▼
OCR per Panel (EasyOCR)
       │
       ▼
HuggingFace Dataset (image + utterance per panel)
       │
       ▼
Vision Model (LLaMA-3.2-Vision or Qwen2.5-VL)
       │
       ├──▶ Emotion Label per Panel
       └──▶ Full-Page Emotion Narrative
```

## Project Structure

```
comics_emorec/
├── app.py                  # Gradio interface
├── ocr/
│   └── extractor.py        # Panel detection (Kumiko) + OCR (EasyOCR)
├── dataset/
│   └── builder.py          # HuggingFace dataset builder
├── models/
│   ├── base_model.py       # Abstract base class
│   ├── llama_vision.py     # LLaMA-3.2-Vision wrapper
│   └── qwen_vl.py          # Qwen2.5-VL wrapper
├── inference/
│   └── pipeline.py         # End-to-end pipeline
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Update model paths in `app.py`:

```python
MODEL_PATHS = {
    "LLaMA-3.2-Vision": "/path/to/llama-3.2-vision-finetuned",
    "Qwen2.5-VL":        "/path/to/qwen2.5-vl-finetuned",
}
```

## Usage

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### On a SLURM cluster

```bash
srun --gres=gpu:1 --pty python app.py
# SSH tunnel from your local machine:
ssh -L 7860:localhost:7860 <cluster>
```

## Models

| Model | Notes |
|---|---|
| **Qwen2.5-VL** | Recommended — stronger OCR and visual understanding |
| **LLaMA-3.2-Vision** | Good baseline, Meta architecture |

## Dataset

Each processed page produces a HuggingFace `Dataset` saved to `./comics_dataset/<page_name>/` with columns:

| Column | Type | Description |
|---|---|---|
| `page_name` | string | Name of the comics page |
| `panel_id` | int32 | Panel index (0-based) |
| `image` | Image | Cropped panel image |
| `utterance` | string | OCR-extracted text from the panel |
