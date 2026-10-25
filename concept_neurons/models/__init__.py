import importlib

MODELS = {
    "qwen2.5-omni-7b": ("concept_neurons.models.qwen", "QwenOmni"),
    "desta2.5-audio": ("concept_neurons.models.desta", "DeSTA25Audio"),
    "audio-flamingo-3": ("concept_neurons.models.audio_flamingo3", "AudioFlamingo3"),
}


def load_model(name: str, device: str = "cuda"):
    """Each wrapper exposes .model, prepare_inputs(audio, text), prepare_text_inputs(text),
    audio_positions(input_ids), text_positions(input_ids, text) and generate(inputs)."""
    module, cls = MODELS[name]
    return getattr(importlib.import_module(module), cls)(device=device)


def find_subsequence(tokenizer, input_ids, text: str) -> list[int]:
    target = tokenizer.encode(text, add_special_tokens=False)
    ids = input_ids[0].tolist()
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    return []
