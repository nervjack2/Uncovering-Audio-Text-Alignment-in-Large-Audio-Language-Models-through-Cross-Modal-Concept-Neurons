"""Pooled FFN activations of every SAKURA audio sample and of the text descriptions of every label.

    python extract_activations.py --model qwen2.5-omni-7b --task emotion

Writes activations/<model>/<task>/audio/<sample id>.npy and audio_index.json (audio side: the question
of the task followed by the audio, pooled over the audio tokens) and text.npz (text side: each
description pooled over its own tokens, then averaged over the five descriptions of a label).
Each array is (layers, neurons per layer), float16.
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from concept_neurons.data import TASKS, SakuraTask
from concept_neurons.hooks import ActivationRecorder
from concept_neurons.models import MODELS, load_model
from concept_neurons.neurons import activation_dir
from concept_neurons.prompts import DESCRIPTIONS, QUESTIONS


def pooled(model, recorder, inputs, positions) -> np.ndarray:
    with torch.inference_mode():
        model.model(**inputs)
    return torch.stack(recorder.pooled(positions)).numpy()


def extract_audio(model, recorder, task: str, out_dir: str) -> None:
    dataset = SakuraTask(task)
    os.makedirs(os.path.join(out_dir, "audio"), exist_ok=True)
    index = []
    for i in tqdm(range(len(dataset)), desc=f"audio/{task}"):
        sample = dataset[i]
        path = os.path.join(out_dir, "audio", f"{sample['id']}.npy")
        if not os.path.isfile(path):
            inputs = model.prepare_inputs(sample["audio"], QUESTIONS[task])
            positions = model.audio_positions(inputs["input_ids"])
            np.save(path, pooled(model, recorder, inputs, positions).astype(np.float16))
        index.append({"id": sample["id"], "label": sample["label"]})
    with open(os.path.join(out_dir, "audio_index.json"), "w") as f:
        json.dump(index, f, indent=1)


def extract_text(model, recorder, descriptions: dict[str, list[str]], path: str) -> None:
    out = {}
    for label, texts in descriptions.items():
        vecs = []
        for text in texts:
            inputs = model.prepare_text_inputs(text)
            vecs.append(pooled(model, recorder, inputs, model.text_positions(inputs["input_ids"], text)))
        out[label] = np.stack(vecs).mean(axis=0).astype(np.float16)
    np.savez_compressed(path, labels=json.dumps(list(out)), **out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--task", choices=list(DESCRIPTIONS), nargs="+", default=list(DESCRIPTIONS))
    ap.add_argument("--activations", default="activations")
    args = ap.parse_args()

    model = load_model(args.model)
    recorder = ActivationRecorder(model.model)
    for task in args.task:
        out_dir = activation_dir(args.activations, args.model, task)
        os.makedirs(out_dir, exist_ok=True)
        descriptions = {label: DESCRIPTIONS[task][label] for label in TASKS[task][1]}
        extract_text(model, recorder, descriptions, os.path.join(out_dir, "text.npz"))
        extract_audio(model, recorder, task, out_dir)


if __name__ == "__main__":
    main()
