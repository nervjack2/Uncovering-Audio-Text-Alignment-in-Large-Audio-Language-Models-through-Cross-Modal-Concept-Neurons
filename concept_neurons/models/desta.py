from concept_neurons.models import find_subsequence
from concept_neurons.models.desta_model import DeSTA2_5Model, Desta2_5Processor

SYSTEM_PROMPT = "Focus on the audio clips and instructions."
AUDIO_PLACEHOLDER = "<|reserved_special_token_87|>"
TRANSCRIPTION_PLACEHOLDER = "<|reserved_special_token_88|>"


class DeSTA25Audio:
    """DeSTA2.5-Audio. Requires the `desta` package (https://github.com/kehanlu/DeSTA2.5-Audio).

    The audio is fed to the backbone as a fixed number of placeholder embeddings, followed by the
    Whisper transcript of the audio. Only the audio placeholders count as audio positions."""

    MODEL_ID = "DeSTA-ntu/DeSTA2.5-Audio-Llama-3.1-8B"

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = DeSTA2_5Model.from_pretrained(self.MODEL_ID).to(device)
        self.model.eval()
        config = {
            "audio_prompt_size": self.model.config.prompt_size,
            "audio_locator": self.model.config.audio_locator,
            "audio_placeholder_token": AUDIO_PLACEHOLDER,
            "transcription_placeholder_token": TRANSCRIPTION_PLACEHOLDER,
        }
        self.processor = Desta2_5Processor(
            text_tokenizer_id=self.model.config.llm_model_id,
            speech_processor_id=self.model.config.encoder_model_id,
            processing_config=config,
        )
        self.tokenizer = self.processor.tokenizer
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(AUDIO_PLACEHOLDER)
        config["audio_placeholder_token_id"] = self.audio_token_id
        config["transcription_placeholder_token_id"] = self.tokenizer.convert_tokens_to_ids(TRANSCRIPTION_PLACEHOLDER)
        self.model.set_processing_config(config)

    def prepare_inputs(self, audio, text: str):
        transcription = self.model.prepare_transcriptions([audio], [None])[0]
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{text}\n<|AUDIO|>", "audios": [{"audio": audio, "text": transcription}]},
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        return self.processor(audio=[audio], transcription=[transcription], text=[prompt],
                              add_special_tokens=False, return_tensors="pt", padding=True).to(self.device)

    def prepare_text_inputs(self, text: str):
        conversation = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)

    def audio_positions(self, input_ids) -> list[int]:
        return (input_ids[0] == self.audio_token_id).nonzero(as_tuple=True)[0].tolist()

    def text_positions(self, input_ids, text: str) -> list[int]:
        return find_subsequence(self.tokenizer, input_ids, text)

    def generate(self, inputs, max_new_tokens: int = 32) -> str:
        # generation runs on input embeddings, so the output contains only the new tokens
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
