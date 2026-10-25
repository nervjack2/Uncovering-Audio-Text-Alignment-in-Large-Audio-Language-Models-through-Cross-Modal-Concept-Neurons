"""Contrastive neuron steering with a text-derived direction (paper, Sec. 5.3).

    python steer.py --model qwen2.5-omni-7b --task emotion --label happy

While the model hears audio of a carrier label c_r, the cross-modal concept neurons of the target
label c_t are shifted by coef * (mu_text(c_t) - mu_text(c_r)). Carriers are test samples of every
other label whose answer options include c_t (up to 20 per label), plus silence and noise. The same
is done with five random neuron sets of the same size. k defaults to the value used in the paper
(configs/steering_k.json). Writes results/steering/<model>_<task>_<label>.json and .png.
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from concept_neurons import neurons as cn
from concept_neurons import plots
from concept_neurons.data import TASKS, SakuraTask, file_id
from concept_neurons.evaluation import option_letter, target_rate
from concept_neurons.hooks import ActivationRecorder, NeuronIntervention
from concept_neurons.models import MODELS, load_model
from concept_neurons.prompts import CARRIER_DESCRIPTIONS

COEFS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def carrier_samples(dataset: SakuraTask, test_ids: dict, target: str, per_carrier: int, seed: int = 0) -> dict:
    """Test samples of each other label whose options include `target`, most available first."""
    candidates = {}
    for label in dataset.labels:
        if label == target:
            continue
        ids = set(test_ids.get(label, []))
        candidates[label] = [i for i, x in enumerate(dataset.info) if x["label"] == label
                             and file_id(x["audio_path"]) in ids and option_letter(x["mc_question"], target)]
    carriers = {}
    for label, idx in sorted(candidates.items(), key=lambda kv: len(kv[1]), reverse=True):
        if idx:
            random.Random(seed).shuffle(idx)
            carriers[label] = [dataset[i] for i in idx[:per_carrier]]
    base = next(iter(carriers.values()))
    carriers["silence"] = [{**s, "audio": np.zeros(len(s["audio"]), dtype=np.float32)} for s in base]
    carriers["noise"] = [{**s, "audio": (np.random.default_rng(seed + i).standard_normal(len(s["audio"])) * 0.05)
                          .astype(np.float32)} for i, s in enumerate(base)]
    return carriers


def text_activation(model, texts: list[str]) -> np.ndarray:
    recorder = ActivationRecorder(model.model)
    vecs = []
    for text in texts:
        inputs = model.prepare_text_inputs(text)
        with torch.inference_mode():
            model.model(**inputs)
        vecs.append(torch.stack(recorder.pooled(model.text_positions(inputs["input_ids"], text))).numpy())
    recorder.remove()
    return np.stack(vecs).mean(axis=0)


def direction(neurons, text: dict, target: str, carrier: str) -> dict:
    diff = text[target] - text[carrier]
    return {layer: (idx, [float(diff[layer, i]) for i in idx]) for layer, idx in cn.by_layer(neurons).items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--k", type=int, help="neuron-set size k (default: configs/steering_k.json)")
    ap.add_argument("--per_carrier", type=int, default=20)
    ap.add_argument("--activations", default="activations")
    ap.add_argument("--out", default="results/steering")
    args = ap.parse_args()

    k = args.k or json.load(open(os.path.join(os.path.dirname(__file__), "configs", "steering_k.json"))
                            )[args.model][args.task][args.label]
    d = cn.load_task(args.activations, args.model, args.task)
    n_layers, dim = d["text"][args.label].shape
    concept = cn.concept_neurons(d["val"], d["text"], [args.label], k)[args.label]
    random_sets = cn.steering_random_sets(len(concept), n_layers, dim, concept, args.model, args.task, args.label)
    print(f"k={k}: {len(concept)} cross-modal concept neurons")

    model = load_model(args.model)
    carriers = carrier_samples(SakuraTask(args.task), d["test_ids"], args.label, args.per_carrier)
    text = dict(d["text"])
    for name, texts in CARRIER_DESCRIPTIONS.items():
        text[name] = text_activation(model, texts)

    hook = NeuronIntervention(model.model)
    result = {"model": args.model, "task": args.task, "label": args.label, "k": k, "N": len(concept),
              "coefs": [0.0] + COEFS, "n_samples": {c: len(s) for c, s in carriers.items()},
              "baseline": {c: target_rate(model, s, args.label) for c, s in carriers.items()}}

    def sweep(neurons) -> dict:
        rates = {}
        for carrier, samples in carriers.items():
            hook.add(direction(neurons, text, args.label, carrier))
            rates[carrier] = {}
            for coef in COEFS:
                hook.coef = coef
                rates[carrier][str(coef)] = target_rate(model, samples, args.label)
            hook.clear()
        return rates

    result["concept"] = sweep(concept)
    result["random"] = [sweep(s) for s in random_sets]

    os.makedirs(args.out, exist_ok=True)
    base = os.path.join(args.out, f"{args.model}_{args.task}_{args.label}")
    with open(base + ".json", "w") as f:
        json.dump(result, f, indent=1)
    plots.steering_curves(result, base + ".png")


if __name__ == "__main__":
    main()
