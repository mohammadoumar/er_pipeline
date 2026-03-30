import cv2
import easyocr
import numpy as np
from PIL import Image


class ComicsOCR:
    def __init__(self, languages: list[str] = ["en"]):
        self.reader = easyocr.Reader(languages, gpu=True)

    def detect_panels(self, image: Image.Image) -> list[Image.Image]:
        """
        Detect comic panels using OpenCV contour detection.
        Finds large rectangular regions separated by thick borders.
        Falls back to the full image if no panels are found.
        """
        img = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_area = img.shape[0] * img.shape[1]
        min_area = img_area * 0.03  # panel must be at least 3% of page

        panels = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                cropped = image.crop((x, y, x + w, y + h))
                panels.append((y, x, cropped))  # sort top-to-bottom, left-to-right

        panels.sort(key=lambda t: (t[0], t[1]))
        panels = [p[2] for p in panels]

        return panels if panels else [image]

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
