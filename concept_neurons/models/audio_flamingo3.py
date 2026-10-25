from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from concept_neurons.models import find_subsequence


class AudioFlamingo3:
    """Audio Flamingo 3 (needs transformers >= 5.0)."""

    MODEL_ID = "nvidia/audio-flamingo-3-hf"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.MODEL_ID, torch_dtype="auto", device_map=device)
        self.model.eval()

    def _inputs(self, content: list):
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, return_dict=True,
        ).to(self.device)

    def prepare_inputs(self, audio, text: str):
        return self._inputs([{"type": "text", "text": text}, {"type": "audio", "audio": audio}])

    def prepare_text_inputs(self, text: str):
        return self._inputs([{"type": "text", "text": text}])

    def audio_positions(self, input_ids) -> list[int]:
        return (input_ids[0] == self.processor.audio_token_id).nonzero(as_tuple=True)[0].tolist()

    def text_positions(self, input_ids, text: str) -> list[int]:
        return find_subsequence(self.tokenizer, input_ids, text)

    def generate(self, inputs, max_new_tokens: int = 32) -> str:
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
