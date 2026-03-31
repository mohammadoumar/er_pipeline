import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer


DEEPSEEK_OCR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "local_models", "DeepSeek-OCR"
)
DEEPSEEK_OCR_PATH = os.path.abspath(DEEPSEEK_OCR_PATH)

OCR_PROMPT = "<image>\nFree OCR. "


class ComicsOCR:
    def __init__(self, model_path: str = DEEPSEEK_OCR_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        try:
            self.model = AutoModel.from_pretrained(
                model_path,
                _attn_implementation="flash_attention_2",
                trust_remote_code=True,
                use_safetensors=True,
            )
        except Exception:
            # Fall back to eager attention if flash-attn is not installed
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_safetensors=True,
            )
        self.model = self.model.eval().cuda().to(torch.bfloat16)

    # ------------------------------------------------------------------
    # Panel detection
    # ------------------------------------------------------------------

    def detect_panels(self, image: Image.Image) -> list[Image.Image]:
        """
        Detect comic panels using OpenCV contour detection.
        Falls back to the full image if no panels are found.
        """
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = img.shape[0] * img.shape[1]
        min_area = img_area * 0.03

        panels = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                panels.append((y, x, image.crop((x, y, x + w, y + h))))

        panels.sort(key=lambda t: (t[0], t[1]))
        panels = [p[2] for p in panels]

        return panels if panels else [image]

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def extract_text(self, image: Image.Image) -> str:
        """Run DeepSeek-OCR on a single image and return extracted text."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        try:
            with tempfile.TemporaryDirectory() as out_dir:
                result = self.model.infer(
                    self.tokenizer,
                    prompt=OCR_PROMPT,
                    image_file=tmp_path,
                    output_path=out_dir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False,
                )
        finally:
            os.unlink(tmp_path)

        return result.strip() if isinstance(result, str) else str(result).strip()

    # ------------------------------------------------------------------
    # Full page processing
    # ------------------------------------------------------------------

    def process_page(self, image: Image.Image) -> list[dict]:
        """
        Detect panels and extract utterances from each.
        Returns a list of dicts: {panel_id, image, utterance}
        """
        panels = self.detect_panels(image)
        return [
            {
                "panel_id": i,
                "image": panel,
                "utterance": self.extract_text(panel),
            }
            for i, panel in enumerate(panels)
        ]

    def process_pages(self, images: list[Image.Image]) -> list[dict]:
        """
        Process multiple pages.
        Returns a list of dicts: {page_id, panel_id, image, utterance}
        """
        results = []
        for page_id, image in enumerate(images):
            for panel in self.process_page(image):
                results.append({
                    "page_id": page_id,
                    **panel,
                })
        return results
