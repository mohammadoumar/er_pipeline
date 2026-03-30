import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

from models.base_model import BaseVisionModel


EMOTION_PROMPT = (
    "You are an emotion recognition system for comics. "
    "Given the panel image and the utterance spoken in the panel, "
    "identify the primary emotion expressed. "
    "Respond with a single emotion label (e.g. joy, sadness, anger, fear, surprise, disgust, neutral).\n\n"
    "Utterance: {utterance}\n"
    "Emotion:"
)

NARRATIVE_PROMPT = (
    "You are an emotion narrative generator for comics. "
    "Below are the panels of a comics page with their utterances and detected emotions:\n\n"
    "{panel_summary}\n\n"
    "Write a cohesive emotion narrative for this comics page, describing how the emotional "
    "arc develops from panel to panel."
)


class LlamaVisionModel(BaseVisionModel):
    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def predict_panel(self, image: Image.Image, utterance: str) -> str:
        prompt = EMOTION_PROMPT.format(utterance=utterance)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=20)

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded.split("Emotion:")[-1].strip().split("\n")[0].strip()

    def generate_narrative(self, panels: list[dict]) -> str:
        panel_summary = "\n".join(
            f"Panel {p['panel_id'] + 1}: utterance='{p['utterance']}', emotion={p['emotion']}"
            for p in panels
        )
        prompt = NARRATIVE_PROMPT.format(panel_summary=panel_summary)

        # Use first panel image as visual context for the narrative
        image = panels[0]["image"]
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}
        ]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=300)

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        return decoded.split(prompt)[-1].strip()
