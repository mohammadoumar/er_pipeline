import easyocr
import numpy as np
from PIL import Image
from kumiko import Kumiko


class ComicsOCR:
    def __init__(self, languages: list[str] = ["en"]):
        self.reader = easyocr.Reader(languages, gpu=True)

    def detect_panels(self, image: Image.Image) -> list[Image.Image]:
        """Use Kumiko to detect and crop individual panels from a comics page."""
        img_array = np.array(image)
        kumiko = Kumiko()
        info = kumiko.parse_images(img_array)

        panels = []
        for panel in info[0].get("panels", []):
            x, y, w, h = panel["x"], panel["y"], panel["w"], panel["h"]
            cropped = image.crop((x, y, x + w, y + h))
            panels.append(cropped)

        # Fallback: treat entire image as one panel if none detected
        if not panels:
            panels = [image]

        return panels

    def extract_text(self, panel: Image.Image) -> str:
        """Run EasyOCR on a single panel and return concatenated text."""
        img_array = np.array(panel)
        results = self.reader.readtext(img_array, detail=0, paragraph=True)
        return " ".join(results).strip()

    def process_page(self, image: Image.Image) -> list[dict]:
        """
        Full pipeline: detect panels, extract text from each.
        Returns a list of dicts: {panel_id, image, utterance}
        """
        panels = self.detect_panels(image)
        output = []
        for i, panel in enumerate(panels):
            utterance = self.extract_text(panel)
            output.append({
                "panel_id": i,
                "image": panel,
                "utterance": utterance,
            })
        return output
