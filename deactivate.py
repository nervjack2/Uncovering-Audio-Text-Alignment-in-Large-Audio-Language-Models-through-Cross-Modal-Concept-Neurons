"""Task-level deactivation of cross-modal concept neurons (paper, Sec. 5.4).

    python deactivate.py --model qwen2.5-omni-7b --task emotion

For each k, the union over the task's labels of the cross-modal concept neurons (selected on the
validation split) is set to zero, and accuracy is measured on the task's test split (target) and on
the other three tasks (controls). Five random sets of the same size are deactivated as a control,
and with --audio_only also a same-size set chosen from the audio rankings alone.
Writes results/deactivation/<model>_<task>.json; finished values of k are skipped on a rerun.
"""
import argparse
import json
import os

from concept_neurons import neurons as cn
from concept_neurons.data import TASKS, SakuraTask, file_id
from concept_neurons.evaluation import accuracy
from concept_neurons.hooks import NeuronIntervention
from concept_neurons.models import MODELS, load_model

K_GRID = [25, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--k", type=int, nargs="+", default=K_GRID)
    ap.add_argument("--random_trials", type=int, default=5)
    ap.add_argument("--audio_only", action="store_true")
    ap.add_argument("--activations", default="activations")
    ap.add_argument("--out", default="results/deactivation")
    args = ap.parse_args()

    d = cn.load_task(args.activations, args.model, args.task)
    labels = TASKS[args.task][1]
    n_layers, dim = d["text"][labels[0]].shape

    target = SakuraTask(args.task)
    test_ids = {i for ids in d["test_ids"].values() for i in ids}
    eval_sets = {args.task: (target, [i for i, x in enumerate(target.info) if file_id(x["audio_path"]) in test_ids])}
    for t in TASKS:
        if t != args.task:
            ds = SakuraTask(t)
            eval_sets[t] = (ds, list(range(len(ds))))

    model = load_model(args.model)
    hook = NeuronIntervention(model.model)

    def evaluate(neurons) -> dict:
        hook.zero(cn.by_layer(neurons))
        acc = {t: accuracy(model, ds, idx, desc=t) for t, (ds, idx) in eval_sets.items()}
        hook.clear()
        return acc

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.model}_{args.task}.json")
    result = json.load(open(path)) if os.path.isfile(path) else {"model": args.model, "task": args.task, "by_k": {}}
    if "baseline" not in result:
        result["baseline"] = evaluate(set())

    for k in args.k:
        if str(k) in result["by_k"]:
            continue
        neurons = set().union(*cn.concept_neurons(d["val"], d["text"], labels, k).values())
        r = {"N": len(neurons), "concept": evaluate(neurons), "random": []}
        if 2 * len(neurons) <= n_layers * dim:  # otherwise no disjoint random set of the same size exists
            r["random"] = [evaluate(cn.random_neurons(len(neurons), n_layers, dim, neurons, seed=t))
                           for t in range(args.random_trials)]
        if args.audio_only:
            r["audio_only"] = evaluate(cn.audio_only_neurons(d["val"], len(neurons)))
        result["by_k"][str(k)] = r
        print(f"k={k} N={len(neurons)} " + " ".join(f"{t}={a:.3f}" for t, a in r["concept"].items()))
        with open(path, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
