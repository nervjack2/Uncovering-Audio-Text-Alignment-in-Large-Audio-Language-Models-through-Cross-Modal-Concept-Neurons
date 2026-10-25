import json
import os

import datasets
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

CACHE_DIR = os.environ.get("SAKURA_CACHE", "./data/sakura")

TASKS = {
    "emotion": ("SLLM-multi-hop/EmotionQA", ["disgust", "sad", "angry", "fear", "happy"]),
    "animal": ("SLLM-multi-hop/AnimalQA", ["cat", "cow", "crow", "dog", "frog", "hen", "pig", "rooster", "sheep"]),
    "gender": ("SLLM-multi-hop/GenderQA", ["female", "male"]),
    "language": ("SLLM-multi-hop/LanguageQA",
                 ["Chinese", "English", "French", "German", "Italian", "Japanese", "Korean", "Spanish"]),
}


def file_id(audio_path: str) -> str:
    return audio_path.replace(os.sep, "_").replace(".", "_")


class SakuraTask:
    """One SAKURA single-hop task. Audio is resampled to 16 kHz and cached as wav files."""

    def __init__(self, task: str, cache_dir: str = CACHE_DIR):
        self.task = task
        self.hf_name, self.labels = TASKS[task]
        self.root = os.path.join(cache_dir, task)
        info_path = os.path.join(self.root, "data_info.json")
        if not os.path.isfile(info_path):
            self._download(info_path)
        with open(info_path) as f:
            self.info = json.load(f)

    def _download(self, info_path: str) -> None:
        os.makedirs(os.path.join(self.root, "wav"), exist_ok=True)
        info = []
        for x in tqdm(datasets.load_dataset(self.hf_name, split="test"), desc=f"Downloading {self.task}"):
            path = os.path.join(self.root, "wav", x["file"])
            if not os.path.isfile(path):
                wav = librosa.resample(x["audio"]["array"], orig_sr=x["audio"]["sampling_rate"], target_sr=16000)
                wavfile.write(path, 16000, (wav * 32767).astype(np.int16))
            info.append({
                "audio_path": os.path.join("wav", x["file"]),
                "label": x["attribute_label"],
                "mc_question": x["single_instruction"],
                "mc_answer": x["single_answer"].strip()[1],
            })
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def __len__(self) -> int:
        return len(self.info)

    def __getitem__(self, i: int) -> dict:
        item = self.info[i]
        audio, _ = librosa.load(os.path.join(self.root, item["audio_path"]), sr=16000)
        return {"id": file_id(item["audio_path"]), "audio": audio, **item}

    def ids_by_label(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for item in self.info:
            out.setdefault(item["label"], []).append(file_id(item["audio_path"]))
        return out


def split_ids(ids_by_label: dict[str, list[str]], val_ratio: float = 0.2, seed: int = 0):
    """Stratified validation/test split of the sample ids of each label."""
    rng = np.random.default_rng(seed)
    val, test = {}, {}
    for label, ids in ids_by_label.items():
        n = len(ids)
        idx = rng.permutation(n)
        n_val = max(1, min(n - 1, int(round(n * val_ratio)))) if n >= 2 else 0
        if n_val > 0:
            val[label] = [ids[i] for i in idx[:n_val]]
        if n - n_val > 0:
            test[label] = [ids[i] for i in idx[n_val:]]
    return val, test
