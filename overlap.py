"""Audio-text overlap: cosine similarity vs. cross-modal concept neurons (paper, Sec. 5.1 and 5.2).

    python overlap.py --model qwen2.5-omni-7b

For every task, k is tuned on the validation split (mean per-label Jaccard overlap of the audio and
text top-k sets), and all reported numbers use the test split: per-label Jaccard overlap with its
hypergeometric p-value (existence), the label-by-label overlap matrix with its exact
label-permutation p-value (distinctiveness), and the matched-label variance share of the overlap
matrix and of the cosine-similarity matrix. Writes results/overlap/<model>_<task>.json and .png.
"""
import argparse
import json
import os

import numpy as np

from concept_neurons import neurons as cn
from concept_neurons import plots, stats
from concept_neurons.data import TASKS
from concept_neurons.models import MODELS

K_GRID = [25, 50, 100, 200, 500, 1000, 2000, 4000]


def cosine_matrix(audio: dict, text: dict, labels: list[str]) -> np.ndarray:
    a = {c: audio[c].ravel() / (np.linalg.norm(audio[c].ravel()) + 1e-12) for c in labels}
    t = {c: text[c].ravel() / (np.linalg.norm(text[c].ravel()) + 1e-12) for c in labels}
    return np.array([[float(a[i] @ t[j]) for j in labels] for i in labels])


def run(model: str, task: str, root: str) -> dict:
    d = cn.load_task(root, model, task)
    labels = sorted(set(d["test"]) & set(d["text"]))
    n_layers, dim = d["text"][labels[0]].shape
    population = n_layers * dim

    text_sets = {k: cn.top_k_sets(d["text"], k) for k in K_GRID}
    val_sets = {k: cn.top_k_sets(d["val"], k) for k in K_GRID}
    val_jaccard = {k: float(np.mean([cn.jaccard(val_sets[k][c], text_sets[k][c]) for c in labels]))
                   for k in K_GRID}
    k = max(K_GRID, key=lambda x: val_jaccard[x])

    audio, text = cn.top_k_sets(d["test"], k), text_sets[k]
    per_label = {}
    for c in labels:
        overlap = len(audio[c] & text[c])
        per_label[c] = {"jaccard": cn.jaccard(audio[c], text[c]),
                        "p_existence": stats.hypergeom_p(overlap, k, k, population)}
    matrix = np.array([[cn.jaccard(audio[i], text[j]) for j in labels] for i in labels])
    cosine = cosine_matrix(d["test"], d["text"], labels)
    off = ~np.eye(len(labels), dtype=bool)
    return {
        "model": model, "task": task, "labels": labels, "k": k, "val_jaccard_by_k": val_jaccard,
        "mean_jaccard": float(np.mean([v["jaccard"] for v in per_label.values()])),
        "random_jaccard": stats.random_jaccard(k, k, population),
        "max_p_existence": max(v["p_existence"] for v in per_label.values()),
        "p_distinctiveness": stats.permutation_test(matrix) if len(labels) > 2 else None,
        "matched_mean": float(np.diag(matrix).mean()), "mismatched_mean": float(matrix[off].mean()),
        "share_neuron": stats.matched_label_share(matrix), "share_cosine": stats.matched_label_share(cosine),
        "per_label": per_label, "jaccard_matrix": matrix.tolist(), "cosine_matrix": cosine.tolist(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--task", choices=list(TASKS), nargs="+", default=list(TASKS))
    ap.add_argument("--activations", default="activations")
    ap.add_argument("--out", default="results/overlap")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"{'task':<9} {'k':>5} {'Jaccard':>8} {'random':>9} {'max p (exist.)':>15} {'perm. p':>8} "
          f"{'share (cos.)':>12} {'share (neuron)':>14}")
    for task in args.task:
        r = run(args.model, task, args.activations)
        base = os.path.join(args.out, f"{args.model}_{task}")
        with open(base + ".json", "w") as f:
            json.dump(r, f, indent=1)
        plots.overlap_matrix(np.array(r["jaccard_matrix"]), r["labels"], task, base + ".png")
        perm = "--" if r["p_distinctiveness"] is None else f"{r['p_distinctiveness']:.2g}"
        print(f"{task:<9} {r['k']:>5} {r['mean_jaccard']:>8.3f} {r['random_jaccard']:>9.6f} "
              f"{r['max_p_existence']:>15.2g} {perm:>8} {r['share_cosine']:>12.2%} {r['share_neuron']:>14.2%}")


if __name__ == "__main__":
    main()
