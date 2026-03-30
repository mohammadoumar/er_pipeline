from PIL import Image

from ocr.extractor import ComicsOCR
from dataset.builder import DatasetBuilder
from models.base_model import BaseVisionModel


class EmotionPipeline:
    def __init__(self, model: BaseVisionModel, dataset_save_dir: str = "./comics_dataset"):
        self.ocr = ComicsOCR()
        self.dataset_builder = DatasetBuilder(save_dir=dataset_save_dir)
        self.model = model

    def run(self, image: Image.Image, page_name: str = "page") -> dict:
        """
        Full pipeline:
          1. Detect panels and extract utterances via OCR
          2. Build and save a HuggingFace dataset
          3. Predict emotion label per panel
          4. Generate a full-page emotion narrative

        Returns:
            {
                "panels": [ {panel_id, image, utterance, emotion}, ... ],
                "narrative": str,
                "dataset_path": str,
                "num_panels": int,
            }
        """
        # Step 1: OCR
        panels = self.ocr.process_page(image)

        # Step 2: Build dataset
        dataset, dataset_path = self.dataset_builder.build_and_save(panels, page_name)

        # Step 3: Per-panel emotion prediction
        for panel in panels:
            panel["emotion"] = self.model.predict_panel(panel["image"], panel["utterance"])

        # Step 4: Full-page narrative
        narrative = self.model.generate_narrative(panels)

        return {
            "panels": panels,
            "narrative": narrative,
            "dataset_path": dataset_path,
            "num_panels": len(panels),
        }
