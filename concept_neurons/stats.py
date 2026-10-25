import hashlib
import itertools

import numpy as np
from scipy import stats


# Overlap (Sec. 5.1, 5.2)

def hypergeom_p(overlap: int, n_a: int, n_b: int, population: int) -> float:
    """P(|A & B| >= overlap) for independent random sets of sizes n_a and n_b."""
    return float(stats.hypergeom.sf(overlap - 1, population, n_a, n_b))


def random_jaccard(n_a: int, n_b: int, population: int, n_samples: int = 100_000, seed: int = 0) -> float:
    """Expected Jaccard overlap of two independent random sets (Monte Carlo)."""
    x = stats.hypergeom.rvs(population, n_a, n_b, size=n_samples, random_state=np.random.default_rng(seed))
    return float(np.mean(x / (n_a + n_b - x)))


def diag_gap(m: np.ndarray) -> float:
    n = len(m)
    return np.trace(m) / n - (m.sum() - np.trace(m)) / (n * n - n)


def permutation_test(m: np.ndarray) -> float:
    """Exact one-sided p-value of the diagonal-minus-off-diagonal gap under all column permutations."""
    n = len(m)
    perms = np.array(list(itertools.permutations(range(n))))
    traces = m[np.arange(n)[None, :], perms].sum(axis=1)
    gaps = traces / n - (m.sum() - traces) / (n * n - n)
    return float(np.mean(gaps >= diag_gap(m) - 1e-12))


def matched_label_share(m: np.ndarray) -> float:
    """Share of the matrix's variance explained by a matched-label (diagonal) term on top of
    row and column offsets (Appendix A.1)."""
    n = len(m)
    return float((n - 1) * diag_gap(m) ** 2 / ((m - m.mean()) ** 2).sum())


# Steering (Sec. 5.3)

def logit(p):
    return np.log(p) - np.log1p(-p)


def prediction_interval_p(t_concept: float, t_random, n_samples: int) -> float:
    """One-sided test that the concept set is above the random sets, on logit(T)."""
    eps = 0.5 / n_samples
    x_o = logit(np.clip(t_concept, eps, 1 - eps))
    x_r = logit(np.clip(np.asarray(t_random, dtype=float), eps, 1 - eps))
    m, sd = len(x_r), np.std(x_r, ddof=1)
    if sd == 0:
        return float("nan")
    return float(stats.t.sf((x_o - x_r.mean()) / (sd * np.sqrt(1 + 1 / m)), m - 1))


def benjamini_hochberg(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    order = np.argsort(pv)
    ranked = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
    out = np.empty(len(pv))
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def rank_of_concept(t_concept: float, t_random, key: str) -> int:
    """Rank of the concept set among all sets by T (1 = highest), ties broken at random."""
    t_r = np.asarray(t_random, dtype=float)
    tied = np.isclose(t_r, t_concept)
    seed = int.from_bytes(hashlib.sha256(f"{key}|rank".encode()).digest()[:8], "little")
    return 1 + int(np.sum((t_r > t_concept) & ~tied)) + int(np.random.default_rng(seed).integers(0, tied.sum() + 1))


def rank_sum_p(ranks: list[int], n_sets: int) -> float:
    """P(sum of ranks <= observed) when every rank is uniform on 1..n_sets."""
    pmf = np.array([1.0])
    for _ in ranks:
        pmf = np.convolve(pmf, np.concatenate([[0.0], np.full(n_sets, 1.0 / n_sets)]))
    return float(min(1.0, pmf[: sum(ranks) + 1].sum()))


# Deactivation (Sec. 5.4)

def decline_onset(n_values: list[int], acc: list[float], baseline: float, rel_drop: float = 0.1):
    """Smallest N from which accuracy stays more than rel_drop below baseline until the end of the sweep."""
    onset = None
    for n, a in reversed(list(zip(n_values, acc))):
        if a >= baseline * (1 - rel_drop):
            break
        onset = n
    return onset
