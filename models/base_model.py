from abc import ABC, abstractmethod
from PIL import Image


class BaseVisionModel(ABC):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.load()

    @abstractmethod
    def load(self):
        """Load model and processor from model_path."""
        pass

    @abstractmethod
    def predict_panel(self, image: Image.Image, utterance: str) -> str:
        """Return an emotion label for a single panel given its image and utterance."""
        pass

    @abstractmethod
    def generate_narrative(self, panels: list[dict]) -> str:
        """
        Generate an emotion narrative for a full page.
        panels: list of dicts with keys: panel_id, image, utterance, emotion
        """
        pass
