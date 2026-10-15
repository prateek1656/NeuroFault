"""Scoring functions shared by every spectral experiment.

Why one module: the verdict files compare numbers across arms (A, B, C'), so
AUC, detection@FPR, captured energy and localization F1 must be defined once.
Conventions that matter for reading a result:

- AUC / detection: higher score = "more likely positive" (faulted/drifted).
  Ties count one half in AUC and are NOT credited in detection@FPR (the
  threshold is a value present in the negatives, detection is strict '>'),
  so quantized statistics cannot inflate detection.
- Everything is float64 and accepts anything torch.as_tensor accepts.
- Bootstrap helpers take an explicit seed and use numpy's PCG64, so a verdict
  built from them is replayable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from neurofault.spectral.types import DTYPE, FaultTruth


def _t(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=DTYPE)


# --------------------------------------------------------------------------- detection


def auc(pos, neg) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie): the Mann-Whitney U statistic / (n_p n_n)."""
    p = _t(pos).flatten()
    q = _t(neg).flatten()
    if p.numel() == 0 or q.numel() == 0:
        raise ValueError("auc needs at least one positive and one negative score")
    greater = (p[:, None] > q[None, :]).sum().item()
    ties = (p[:, None] == q[None, :]).sum().item()
    return float((greater + 0.5 * ties) / (p.numel() * q.numel()))


def detection_at_fpr(pos, neg, fpr: float) -> float:
    """Fraction of positives strictly above the threshold that lets at most `fpr` of
    negatives through. The threshold is the floor(fpr*n)-th largest negative, so the
    realized FPR is <= fpr even with ties (conservative)."""
    if not 0.0 <= fpr <= 1.0:
        raise ValueError("fpr must be in [0, 1]")
    p = _t(pos).flatten()
    q = _t(neg).flatten()
    if p.numel() == 0 or q.numel() == 0:
        raise ValueError("detection_at_fpr needs positives and negatives")
    k = int(np.floor(fpr * q.numel()))
    thr = torch.sort(q, descending=True).values[min(k, q.numel() - 1)]
    return float((p > thr).double().mean().item())


# --------------------------------------------------------------------------- low-rank


def rank_energy(
    E,
    ks: Sequence[int],
    *,
    C_sqrt=None,
    basis=None,
    side: str = "right",
) -> dict[int, float]:
    """Fraction of ||E C^{1/2}||_F^2 removed by a rank-k correction (F0.2).

    E is (m, n); C_sqrt is the (n, n) square root of the input covariance (None = unweighted,
    i.e. plain Frobenius energy). Weighting by real inputs matters because the pilot showed
    unweighted capture hides the fact that inputs themselves sit in few directions.

    basis=None: the oracle - best rank-k approximation of the weighted matrix M = E C^{1/2};
    captured energy = sum of the top-k s_i^2 / total (Eckart-Young; equals the weighted
    optimum because right-multiplying by C^{1/2} is a fixed linear map).
    basis given: columns are orthonormal directions (first k used). side="right" projects
    inputs (E <- E B_k B_k^T, e.g. activation PCs or right singular vectors of W0);
    side="left" projects outputs (E <- B_k B_k^T E). Captured = 1 - ||(E - E_k) Cs||^2/||E Cs||^2.
    """
    E = _t(E)
    Cs = None if C_sqrt is None else _t(C_sqrt)
    M = E if Cs is None else E @ Cs
    total = float((M**2).sum().item())
    out: dict[int, float] = {}
    if total == 0.0:
        return {int(k): 0.0 for k in ks}
    if basis is None:
        s2 = torch.linalg.svdvals(M) ** 2
        cum = torch.cumsum(s2, 0)
        for k in ks:
            kk = min(int(k), s2.numel())
            out[int(k)] = float(cum[kk - 1].item() / total) if kk > 0 else 0.0
        return out
    B = _t(basis)
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")
    for k in ks:
        Bk = B[:, : int(k)]
        Ek = E @ Bk @ Bk.T if side == "right" else Bk @ Bk.T @ E
        R = E - Ek
        R = R if Cs is None else R @ Cs
        out[int(k)] = float(1.0 - (R**2).sum().item() / total)
    return out


def rel_output_error(W_hat, W_ref, X) -> float:
    """||(W_hat - W_ref) X||_F / ||W_ref X||_F; X is (n, p) held-out activations."""
    Wh, Wr, X = _t(W_hat), _t(W_ref), _t(X)
    den = torch.linalg.norm(Wr @ X).item()
    if den == 0.0:
        raise ValueError("reference output is identically zero")
    return float(torch.linalg.norm((Wh - Wr) @ X).item() / den)


def subspace_angle(U, V) -> float:
    """Largest principal angle (radians) between span(U) and span(V). Columns need not be
    orthonormal; both are orthonormalized by QR first. 0 = identical, pi/2 = orthogonal."""
    Qu, _ = torch.linalg.qr(_t(U))
    Qv, _ = torch.linalg.qr(_t(V))
    s = torch.linalg.svdvals(Qu.T @ Qv).clamp(max=1.0)
    return float(torch.acos(s.min()).item())


# --------------------------------------------------------------------------- localization


def _keys(cells, ncols: int) -> set[int]:
    c = torch.as_tensor(cells, dtype=torch.long).reshape(-1, 2)
    return set((c[:, 0] * ncols + c[:, 1]).tolist())


def localization_f1(pred_cells, truth_cells) -> float:
    """Cell-level F1 between predicted and true (row, col) sets. Both empty = 1.0 (a correct
    'no fault' call); exactly one empty = 0.0."""
    p = torch.as_tensor(pred_cells, dtype=torch.long).reshape(-1, 2)
    t = torch.as_tensor(truth_cells, dtype=torch.long).reshape(-1, 2)
    if p.numel() == 0 and t.numel() == 0:
        return 1.0
    if p.numel() == 0 or t.numel() == 0:
        return 0.0
    ncols = int(max(p[:, 1].max().item(), t[:, 1].max().item())) + 1
    ps, ts = _keys(p, ncols), _keys(t, ncols)
    tp = len(ps & ts)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(ps), tp / len(ts)
    return 2 * prec * rec / (prec + rec)


def _jaccard(pred, truth) -> float:
    a = set(torch.as_tensor(pred, dtype=torch.long).flatten().tolist())
    b = set(torch.as_tensor(truth, dtype=torch.long).flatten().tolist())
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def row_accuracy(pred_rows, truth: FaultTruth) -> float:
    """Jaccard of predicted vs true affected rows. For one fault and one predicted row this is
    the 0/1 'row located' score the registered criteria quote; for many faults it also penalizes spurious rows."""
    return _jaccard(pred_rows, truth.rows)


def col_accuracy(pred_cols, truth: FaultTruth) -> float:
    """Column analogue of row_accuracy (the backward-read half of two-stage design)."""
    return _jaccard(pred_cols, truth.cols)


# --------------------------------------------------------------------------- bootstrap


def bootstrap_ci(
    x, *, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(lo, hi, point) percentile bootstrap CI of stat(x) over resampled items (e.g. tiles)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = np.apply_along_axis(stat, 1, a[idx])
    return (
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
        float(stat(a)),
    )


def paired_bootstrap_ci(
    a, b, *, stat=np.mean, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """CI of stat(a - b) resampling PAIRS (same seed/drift realization) - the pre-registered protocol.
    Pairing removes the checkpoint/realization variance common to both arms, which is why the
    plans demand a paired lower bound > 0 rather than overlapping unpaired intervals."""
    da = np.asarray(a, dtype=np.float64).ravel()
    db = np.asarray(b, dtype=np.float64).ravel()
    if da.shape != db.shape:
        raise ValueError("paired bootstrap needs equal-length inputs")
    return bootstrap_ci(da - db, stat=stat, n_boot=n_boot, alpha=alpha, seed=seed)


def paired_ratio_ci(
    num, den, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """CI of mean(num)/mean(den) over resampled pairs - e.g. the C1 read-count ratio
    (bidirectional / forward-only), whose 95% upper bound must be <= 0.8."""
    n_ = np.asarray(num, dtype=np.float64).ravel()
    d_ = np.asarray(den, dtype=np.float64).ravel()
    if n_.shape != d_.shape:
        raise ValueError("paired ratio needs equal-length inputs")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_.size, size=(n_boot, n_.size))
    boots = n_[idx].mean(1) / d_[idx].mean(1)
    return (
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
        float(n_.mean() / d_.mean()),
    )
