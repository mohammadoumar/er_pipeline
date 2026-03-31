from PIL import Image

from dataset.builder import DatasetBuilder
from models.base_model import BaseVisionModel


class EmotionPipeline:
    def __init__(self, model: BaseVisionModel, dataset_save_dir: str = "./comics_dataset"):
        self.dataset_builder = DatasetBuilder(save_dir=dataset_save_dir)
        self.model = model

    def run(self, panels: list[dict], page_name: str = "page") -> dict:
        """
        Run emotion prediction on pre-extracted panels (output of ComicsOCR).

        Args:
            panels: list of dicts with keys: panel_id, image, utterance
            page_name: used as dataset save path

        Returns:
            {panels (with emotion key added), narrative, dataset_path, num_panels}
        """
        # Build and save dataset
        dataset, dataset_path = self.dataset_builder.build_and_save(panels, page_name)

        # Per-panel emotion prediction
        for panel in panels:
            panel["emotion"] = self.model.predict_panel(panel["image"], panel["utterance"])

        # Full-page narrative
        narrative = self.model.generate_narrative(panels)

        return {
            "panels": panels,
            "narrative": narrative,
            "dataset_path": dataset_path,
            "num_panels": len(panels),
        }
