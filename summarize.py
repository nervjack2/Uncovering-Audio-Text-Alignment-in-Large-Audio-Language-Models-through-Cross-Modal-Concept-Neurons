"""Significance tests and figures for the steering and deactivation results.

    python summarize.py

Steering (Table 3): for every label, the concept set's target rate averaged over carriers and lambda
is compared with the five random sets by a prediction-interval t-test on the logit scale, with
Benjamini-Hochberg correction over all labels found, and per model by a rank test.
Deactivation (Table 4): decline onsets of the target and control accuracy curves.
"""
import argparse
import glob
import json
import os

import numpy as np

from concept_neurons import plots, stats
from concept_neurons.data import TASKS

MODEL_ORDER = ["qwen2.5-omni-7b", "desta2.5-audio", "audio-flamingo-3"]


def mean_rate(rates: dict) -> float:
    return float(np.mean([r for carrier in rates.values() for r in carrier.values()]))


def steering(root: str) -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        r = json.load(open(path))
        t_concept, t_random = mean_rate(r["concept"]), [mean_rate(x) for x in r["random"]]
        key = f"{r['model']}|{r['task']}|{r['label']}"
        rows.append({**{k: r[k] for k in ["model", "task", "label", "N"]},
                     "effect": t_concept - float(np.mean(t_random)),
                     "p": stats.prediction_interval_p(t_concept, t_random, sum(r["n_samples"].values())),
                     "rank": stats.rank_of_concept(t_concept, t_random, key), "n_sets": len(t_random) + 1})
    if not rows:
        return
    for row, q in zip(rows, stats.benjamini_hochberg([r["p"] for r in rows])):
        row["significant"] = bool(q < 0.05)

    print(f"Steering: significant labels (BH q < 0.05 over {len(rows)} labels)")
    print(f"{'model':<18}" + "".join(f"{t:>10}" for t in TASKS) + f"{'total':>10}{'mean rank':>11}{'p (rank)':>10}")
    for model in [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]:
        rs = [r for r in rows if r["model"] == model]
        cells = []
        for task in TASKS:
            sub = [r for r in rs if r["task"] == task]
            cells.append(f"{sum(r['significant'] for r in sub)}/{len(sub)}" if sub else "--")
        p_rank = stats.rank_sum_p([r["rank"] for r in rs], rs[0]["n_sets"])
        print(f"{model:<18}" + "".join(f"{c:>10}" for c in cells)
              + f"{sum(r['significant'] for r in rs)}/{len(rs):<3}".rjust(10)
              + f"{np.mean([r['rank'] for r in rs]):>11.2f}{p_rank:>10.2g}")
    print()


def deactivation(root: str, figures: str) -> None:
    by_task: dict = {}
    for path in sorted(glob.glob(os.path.join(root, "*.json"))):
        r = json.load(open(path))
        task, controls = r["task"], [t for t in TASKS if t != r["task"]]
        ks = sorted(r["by_k"], key=int)
        runs = [r["by_k"][k] for k in ks]

        def control_mean(acc: dict) -> float:
            return float(np.mean([acc[t] for t in controls]))

        curve = {
            "N": [x["N"] for x in runs],
            "target": [x["concept"][task] for x in runs],
            "control": [control_mean(x["concept"]) for x in runs],
            "random_target": [[t[task] for t in x["random"]] for x in runs],
            "random_control": [[control_mean(t) for t in x["random"]] for x in runs],
            "baseline_target": r["baseline"][task], "baseline_control": control_mean(r["baseline"]),
        }
        if all("audio_only" in x for x in runs):
            curve["audio_only"] = [x["audio_only"][task] for x in runs]
        by_task.setdefault(task, {})[r["model"]] = curve
    if not by_task:
        return

    print("Deactivation: N at which accuracy starts to decline (>10% relative drop, sustained)")
    print(f"{'task':<9} {'model':<18} {'target N':>10} {'control N':>10}  selective")
    for task in [t for t in TASKS if t in by_task]:
        curves = {m: by_task[task][m] for m in MODEL_ORDER if m in by_task[task]}
        for model, c in curves.items():
            t_on = stats.decline_onset(c["N"], c["target"], c["baseline_target"])
            c_on = stats.decline_onset(c["N"], c["control"], c["baseline_control"])
            selective = t_on is not None and (c_on is None or t_on < c_on)
            print(f"{task:<9} {model:<18} {str(t_on):>10} {str(c_on):>10}  {'yes' if selective else 'no'}")
        plots.deactivation_curves(curves, task, os.path.join(figures, f"deactivation_{task}.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    figures = os.path.join(args.results, "figures")
    os.makedirs(figures, exist_ok=True)
    steering(os.path.join(args.results, "steering"))
    deactivation(os.path.join(args.results, "deactivation"), figures)


if __name__ == "__main__":
    main()
