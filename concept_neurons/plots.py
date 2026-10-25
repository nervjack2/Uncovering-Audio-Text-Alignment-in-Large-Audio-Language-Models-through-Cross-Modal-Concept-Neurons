"""The paper's figures: overlap matrices (Figs. 5-7), steering curves (Figs. 3, 8, 9) and
deactivation curves (Figs. 4, 10-12)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, GREY, ORANGE, YELLOW, RED, PURPLE = "#2a78d6", "#898781", "#eb6834", "#fab219", "#d03b3b", "#4a3aa7"
MODEL_NAMES = {"qwen2.5-omni-7b": "Qwen2.5-Omni", "desta2.5-audio": "DeSTA2.5-Audio",
               "audio-flamingo-3": "Audio Flamingo 3"}


def overlap_matrix(matrix: np.ndarray, labels: list[str], title: str, path: str, fontsize: int = 20) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1), max(7, len(labels) * 1.1)))
    vmax = matrix.max() or 1
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xlabel("Text Label", fontsize=fontsize)
    ax.set_ylabel("Audio Label", fontsize=fontsize)
    ax.set_title(title[:1].upper() + title[1:], fontsize=fontsize + 6)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=fontsize * 0.6)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=max(fontsize - 6, 8),
                    color="white" if matrix[i, j] > vmax * 0.6 else "black")
    plt.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def steering_curves(result: dict, path: str) -> None:
    """One panel per carrier: target rate vs. lambda for the concept set and the random sets."""
    coefs = [c for c in result["coefs"] if c != 0]
    carriers = list(result["baseline"])
    ncols = min(3, len(carriers))
    nrows = -(-len(carriers) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.9 * nrows), sharey=True, squeeze=False)
    for ax in axes.flat[len(carriers):]:
        ax.axis("off")
    for ax, carrier in zip(axes.flat, carriers):
        base = result["baseline"][carrier]
        xs = [0.0] + coefs
        ax.axhline(0.0, color="#d8d5cc", linewidth=0.8, zorder=0)
        ax.axhline(base, color=GREY, linestyle=":", linewidth=2.4, label="baseline (coef=0)")
        concept = result["concept"][carrier]
        ax.plot(xs, [base] + [concept[str(c)] for c in coefs], marker="^", markersize=9, color=BLUE,
                linestyle="--", linewidth=2.6, label="cross-modal concept neuron")
        rand = [[base] * len(result["random"])] + [[r[carrier][str(c)] for r in result["random"]] for c in coefs]
        ax.fill_between(xs, [min(v) for v in rand], [max(v) for v in rand], color=GREY, alpha=0.25, linewidth=0)
        ax.plot(xs, [float(np.median(v)) for v in rand], marker="^", markersize=9, color=GREY,
                linestyle="--", linewidth=2.6, label="random neuron")
        ax.set_xlabel("steering strength (coef)", fontsize=22)
        ax.set_title(f"carrier={carrier}", fontsize=26)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=22)
    for row in range(nrows):
        axes[row, 0].set_ylabel(f"P(predict '{result['label']}')", fontsize=22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=3, fontsize=26, mode="expand",
               bbox_to_anchor=(0.0, 1.0 + 0.1 / nrows, 1.0, 0.001), handlelength=3.2, handletextpad=0.6)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def deactivation_curves(curves: dict[str, dict], task: str, path: str, rel_drop: float = 0.1) -> None:
    """One panel per model: accuracy of the task and of the control tasks vs. the number of
    deactivated neurons N, with the random-neuron and audio-only controls."""
    from concept_neurons.stats import decline_onset

    fig, axes = plt.subplots(1, len(curves), figsize=(6.2 * len(curves), 5.8), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (model, c) in zip(axes, curves.items()):
        n = c["N"]
        ax.plot(n, c["target"], marker="o", color=BLUE, linewidth=2, markersize=5,
                label=f"{task} (cross-modal concept neurons)", zorder=3)
        ax.plot(n, c["control"], marker="s", color=ORANGE, linewidth=2, markersize=5,
                label="control tasks (cross-modal concept neurons)", zorder=3)
        ax.axhline(c["baseline_target"], color=BLUE, linestyle=":", linewidth=2.4, alpha=0.8)
        ax.axhline(c["baseline_control"], color=ORANGE, linestyle=":", linewidth=2.4, alpha=0.8)
        rn = [x for x, t in zip(n, c["random_target"]) if t]
        trials = np.array([t for t in c["random_target"] if t])
        ax.fill_between(rn, trials.min(1), trials.max(1), color=BLUE, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(rn, np.median(trials, 1), color=BLUE, linewidth=1.3, linestyle="--", alpha=0.55, zorder=2,
                label=f"{task} (random neuron, median and range of 5 trials)")
        ax.plot(rn, [np.mean(t) for t in c["random_control"] if t], color=ORANGE, linewidth=1.3, linestyle="--",
                alpha=0.55, zorder=2, label="control tasks (random neuron, mean of 5 trials)")
        t_on = decline_onset(n, c["target"], c["baseline_target"], rel_drop)
        c_on = decline_onset(n, c["control"], c["baseline_control"], rel_drop)
        if t_on is not None:
            ax.axvline(t_on, color=YELLOW, linestyle="--", linewidth=1.5, alpha=0.9, zorder=1)
        if c_on is not None:
            ax.axvline(c_on, color=RED, linestyle="--", linewidth=1.5, alpha=0.9, zorder=1)
        if t_on is not None and c_on is not None and c_on > t_on:
            ax.axvspan(t_on, c_on, color=YELLOW, alpha=0.06, zorder=0)
        if c.get("audio_only"):
            ax.plot(n, c["audio_only"], marker="*", markersize=16, color=PURPLE, linestyle=":", linewidth=1.3,
                    alpha=0.9, zorder=4, label=f"{task} (audio-only neurons)")
        ax.set_xscale("log")
        ax.set_xlabel("number of deactivated neurons $N$", fontsize=22)
        ax.set_title(MODEL_NAMES.get(model, model), fontsize=26, pad=14)
        ax.grid(True, alpha=0.3, color="#e1e0d9")
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(axis="both", labelsize=22)
    axes[0].set_ylabel("accuracy", fontsize=22)
    legend = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            legend.setdefault(l, h)
    legend[f"target starts declining (>{rel_drop:.0%} rel. drop)"] = Line2D([], [], color=YELLOW, linestyle="--")
    legend[f"control starts declining (>{rel_drop:.0%} rel. drop)"] = Line2D([], [], color=RED, linestyle="--")
    fig.legend(legend.values(), legend.keys(), loc="upper center", ncol=3, fontsize=18, bbox_to_anchor=(0.5, 1.05))
    fig.subplots_adjust(top=0.64, wspace=0.15, left=0.045, right=0.995)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
