import torch
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

from concept_neurons.models import find_subsequence

SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
                 "perceiving auditory and visual inputs, as well as generating text and speech.")


class QwenOmni:
    MODEL_ID = "Qwen/Qwen2.5-Omni-7B"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = Qwen2_5OmniProcessor.from_pretrained(self.MODEL_ID)
        self.tokenizer = self.processor.tokenizer
        self.audio_bos = self.tokenizer.convert_tokens_to_ids(self.processor.audio_bos_token)
        self.audio_eos = self.tokenizer.convert_tokens_to_ids(self.processor.audio_eos_token)
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval()

    def _chat(self, content: list) -> str:
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        return self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    def prepare_inputs(self, audio, text: str):
        prompt = self._chat([{"type": "text", "text": text}, {"type": "audio", "audio_url": "x"}])
        return self.processor(audio=[audio], text=[prompt], return_tensors="pt", sampling_rate=16000).to(self.device)

    def prepare_text_inputs(self, text: str):
        prompt = self._chat([{"type": "text", "text": text}])
        return self.processor(text=[prompt], return_tensors="pt").to(self.device)

    def audio_positions(self, input_ids) -> list[int]:
        ids = input_ids[0]
        start = (ids == self.audio_bos).nonzero(as_tuple=True)[0]
        end = (ids == self.audio_eos).nonzero(as_tuple=True)[0]
        if len(start) == 0 or len(end) == 0:
            return []
        return list(range(start[0].item() + 1, end[0].item()))

    def text_positions(self, input_ids, text: str) -> list[int]:
        return find_subsequence(self.tokenizer, input_ids, text)

    def generate(self, inputs, max_new_tokens: int = 32) -> str:
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id, bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
