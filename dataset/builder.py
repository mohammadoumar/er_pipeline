import os
from datasets import Dataset, Features, Value, Image as HFImage


class DatasetBuilder:
    def __init__(self, save_dir: str = "./comics_dataset"):
        self.save_dir = save_dir

    def build(self, panels: list[dict], page_name: str = "page") -> Dataset:
        """
        Build a HuggingFace Dataset from a list of panel dicts.

        Each panel dict must have:
            - panel_id (int)
            - image (PIL.Image)
            - utterance (str)

        Returns a HuggingFace Dataset with columns:
            page_name, panel_id, image, utterance
        """
        data = {
            "page_name": [page_name] * len(panels),
            "panel_id": [p["panel_id"] for p in panels],
            "image": [p["image"] for p in panels],
            "utterance": [p["utterance"] for p in panels],
        }

        features = Features({
            "page_name": Value("string"),
            "panel_id": Value("int32"),
            "image": HFImage(),
            "utterance": Value("string"),
        })

        dataset = Dataset.from_dict(data, features=features)
        return dataset

    def save(self, dataset: Dataset, page_name: str = "page") -> str:
        """Save dataset to disk and return the saved path."""
        save_path = os.path.join(self.save_dir, page_name)
        os.makedirs(save_path, exist_ok=True)
        dataset.save_to_disk(save_path)
        return save_path

    def build_and_save(self, panels: list[dict], page_name: str = "page") -> tuple[Dataset, str]:
        dataset = self.build(panels, page_name)
        path = self.save(dataset, page_name)
        return dataset, path
