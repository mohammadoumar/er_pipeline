import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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


class QwenVLModel(BaseVisionModel):
    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    def _build_inputs(self, image: Image.Image, text: str):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]}
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model.device)
        return inputs

    def predict_panel(self, image: Image.Image, utterance: str) -> str:
        prompt = EMOTION_PROMPT.format(utterance=utterance)
        inputs = self._build_inputs(image, prompt)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=20)

        decoded = self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.split("Emotion:")[-1].strip().split("\n")[0].strip()

    def generate_narrative(self, panels: list[dict]) -> str:
        panel_summary = "\n".join(
            f"Panel {p['panel_id'] + 1}: utterance='{p['utterance']}', emotion={p['emotion']}"
            for p in panels
        )
        prompt = NARRATIVE_PROMPT.format(panel_summary=panel_summary)
        image = panels[0]["image"]
        inputs = self._build_inputs(image, prompt)

        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=300)

        decoded = self.processor.batch_decode(
            output, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return decoded.split(prompt)[-1].strip()
