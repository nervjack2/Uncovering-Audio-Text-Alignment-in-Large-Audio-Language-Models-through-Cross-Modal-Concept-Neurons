"""Cross-modal concept neurons (paper, Sec. 3).

A neuron is a pair (layer, index). Activations of one input are stored as an (L, D) array
(one pooled vector per FFN layer); label-level activations are dicts {label: (L, D) array}."""
import hashlib
import json
import os

import numpy as np

from concept_neurons.data import split_ids

Neuron = tuple[int, int]


# Loading

def activation_dir(root: str, model: str, task: str) -> str:
    return os.path.join(root, model, task)


def load_audio(root: str, model: str, task: str) -> dict[str, list[tuple[str, np.ndarray]]]:
    """{label: [(sample id, (L, D) float32 array), ...]} in dataset order."""
    d = activation_dir(root, model, task)
    with open(os.path.join(d, "audio_index.json")) as f:
        index = json.load(f)
    out: dict[str, list] = {}
    for item in index:
        acts = np.load(os.path.join(d, "audio", f"{item['id']}.npy")).astype(np.float32)
        out.setdefault(item["label"], []).append((item["id"], acts))
    return out


def load_text(root: str, model: str, task: str) -> dict[str, np.ndarray]:
    with np.load(os.path.join(activation_dir(root, model, task), "text.npz")) as z:
        labels = json.loads(str(z["labels"]))
        return {label: z[label].astype(np.float32) for label in labels}


def mean_by_label(samples: dict[str, list[tuple[str, np.ndarray]]], ids: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Average the samples of each label whose id is in `ids` (e.g. one split)."""
    out = {}
    for label, items in samples.items():
        keep = set(ids.get(label, []))
        vecs = [a for i, a in items if i in keep]
        if vecs:
            out[label] = np.stack(vecs).mean(axis=0)
    return out


def load_task(root: str, model: str, task: str, val_ratio: float = 0.2, seed: int = 0) -> dict:
    """Label means of the validation and test audio, the text activations and the split itself."""
    samples = load_audio(root, model, task)
    val_ids, test_ids = split_ids({c: [i for i, _ in items] for c, items in samples.items()}, val_ratio, seed)
    return {"val": mean_by_label(samples, val_ids), "test": mean_by_label(samples, test_ids),
            "text": load_text(root, model, task), "val_ids": val_ids, "test_ids": test_ids}


# Selection

def distinctiveness(acts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """s(c) = mu(c) - mean of mu(c') over the other labels c'."""
    out = {}
    for label, vec in acts.items():
        others = [acts[l] for l in acts if l != label]
        out[label] = vec - np.stack(others).mean(axis=0) if others else vec.copy()
    return out


def ranking(scores: np.ndarray) -> np.ndarray:
    """Flat neuron indices sorted by descending score (ties: higher layer, then higher index first)."""
    flat = scores.ravel()
    return np.lexsort((np.arange(flat.size), flat))[::-1]


def top_k(scores: np.ndarray, k: int) -> set[Neuron]:
    d = scores.shape[1]
    return {(int(f // d), int(f % d)) for f in ranking(scores)[:k]}


def top_k_sets(acts: dict[str, np.ndarray], k: int) -> dict[str, set[Neuron]]:
    return {label: top_k(s, k) for label, s in distinctiveness(acts).items()}


def concept_neurons(audio_acts: dict, text_acts: dict, labels: list[str], k: int) -> dict[str, set[Neuron]]:
    """TopK_audio(c) & TopK_text(c) for every label c."""
    audio, text = top_k_sets(audio_acts, k), top_k_sets(text_acts, k)
    return {c: audio.get(c, set()) & text.get(c, set()) for c in labels}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def by_layer(neurons) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer, i in sorted(neurons):
        out.setdefault(layer, []).append(i)
    return out


# Controls

def random_neurons(n: int, n_layers: int, dim: int, exclude: set[Neuron], seed: int) -> set[Neuron]:
    """n distinct random neurons outside `exclude` (deactivation control)."""
    rng = np.random.default_rng(seed)
    chosen: set[Neuron] = set()
    while len(chosen) < n:
        pair = (int(rng.integers(0, n_layers)), int(rng.integers(0, dim)))
        if pair not in exclude and pair not in chosen:
            chosen.add(pair)
    return chosen


def steering_random_sets(n: int, n_layers: int, dim: int, exclude: set[Neuron],
                         model: str, task: str, label: str, n_sets: int = 5) -> list[set[Neuron]]:
    """n_sets random sets of n neurons drawn uniformly without replacement outside `exclude`,
    seeded by (model, task, label, set index)."""
    excl = np.array(sorted(l * dim + i for l, i in exclude), dtype=np.int64)
    pool = n_layers * dim - len(excl)
    sets = []
    for m in range(n_sets):
        seed = int.from_bytes(hashlib.sha256(f"{model}|{task}|{label}|{m}".encode()).digest()[:8], "little")
        pos = np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False).astype(np.int64))
        flat = pos + np.searchsorted(excl - np.arange(len(excl)), pos, side="right")
        sets.append({(int(f // dim), int(f % dim)) for f in flat})
    return sets


def audio_only_neurons(audio_acts: dict[str, np.ndarray], n: int) -> set[Neuron]:
    """n neurons from the audio rankings alone: each label first gets its top n // |C|, then labels take
    turns adding their next-ranked unused neuron until the union has n neurons."""
    labels = list(audio_acts)
    scores = distinctiveness(audio_acts)
    d = next(iter(scores.values())).shape[1]
    ranked = {c: ranking(scores[c]) for c in labels}
    base = n // len(labels)
    chosen = {(int(f // d), int(f % d)) for c in labels for f in ranked[c][:base]}
    nxt = {c: base for c in labels}
    while len(chosen) < n:
        for c in labels:
            if len(chosen) >= n:
                break
            while nxt[c] < len(ranked[c]):
                f = ranked[c][nxt[c]]
                cand = (int(f // d), int(f % d))
                nxt[c] += 1
                if cand not in chosen:
                    chosen.add(cand)
                    break
    return chosen
