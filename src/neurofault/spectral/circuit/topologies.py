"""Loop topologies: which conductance matrices the op-amp nodes actually see.

Every topology is reduced to the same three objects so one ODE integrator serves
all of them:

* ``mp`` / ``mn`` - non-negative (N, N) matrices of the cells that are driven by
  the inverted output (-o, "positive" array) and by the doubly inverted output
  (+o, "negative" array). The ideal-inverter loop matrix is ``M = mp - mn``.
  Keeping them separate is what lets the inverter-pole option delay the two
  paths differently, and what makes G+/G- signing physical (an array can only
  hold non-negative conductances).
* ``load`` - the row sums S_i of every conductance that hangs on node i. This is
  what slows the loop (S/sigma_1 ~ 8-10 for differential pairs) and is the S in
  the growth-rate formula of ``loop.py``.
* ``base_eff`` - the signed matrix whose dominant eigen/singular pair the loop
  should converge to (the "expected fixed point").

All arrays are float64 numpy; the integrator is numpy because the per-step work
is a small matvec and torch call overhead would dominate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class Topology:
    kind: str
    mp: np.ndarray
    mn: np.ndarray
    load: np.ndarray
    n_u: int  # amps on the A-side (rows); for square_signed n_u = N, n_v = 0
    n_v: int
    base_eff: np.ndarray  # signed matrix defining the reference eigenpair
    ref_kind: str  # "eig" | "svd_dilation"
    notes: dict = field(default_factory=dict)

    @property
    def n_amps(self) -> int:
        return self.n_u + self.n_v

    @property
    def m_eff(self) -> np.ndarray:
        return self.mp - self.mn


def _np(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _split(g_pos, g_neg) -> tuple[np.ndarray, np.ndarray]:
    """(G+, G-) non-negative arrays. g_neg None means g_pos is a signed matrix
    realized ideally (ConductanceState 'signed_ideal'): split by sign."""
    gp = _np(g_pos)
    if g_neg is None:
        return np.clip(gp, 0.0, None), np.clip(-gp, 0.0, None)
    return gp, _np(g_neg)


def _dilate(a: np.ndarray) -> np.ndarray:
    m, n = a.shape
    out = np.zeros((m + n, m + n))
    out[:m, m:] = a
    out[m:, :m] = a.T
    return out


def square_signed(g_pos, g_neg) -> Topology:
    """Sun & Ielmini eigenvector loop on a square matrix: node i sees row i of both
    arrays, driven by the inverted outputs of all nodes. Needs real dominant
    eigenvalue (positive or symmetric matrices)."""
    gp, gn = _split(g_pos, g_neg)
    if gp.shape[0] != gp.shape[1]:
        raise ValueError("square_signed needs a square matrix; use dilated_two_array")
    return Topology(
        kind="square_signed",
        mp=gp,
        mn=gn,
        load=(gp + gn).sum(1),
        n_u=gp.shape[0],
        n_v=0,
        base_eff=gp - gn,
        ref_kind="eig",
    )


def dilated_two_array(g_pos, g_neg) -> Topology:
    """[[0, A], [A^T, 0]] realized with two arrays of the same A (the only
    physically supported construction for non-symmetric/rectangular tiles).
    u-amps load on row sums, v-amps on column sums."""
    gp, gn = _split(g_pos, g_neg)
    m, n = gp.shape
    tot = gp + gn
    return Topology(
        kind="dilated_two_array",
        mp=_dilate(gp),
        mn=_dilate(gn),
        load=np.concatenate([tot.sum(1), tot.sum(0)]),
        n_u=m,
        n_v=n,
        base_eff=gp - gn,
        ref_kind="svd_dilation",
    )


def pilot(
    g_pos,
    g_neg,
    beta: float,
    u0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    *,
    rng: np.random.Generator | None = None,
    loads_node: bool = False,
    s1_ref: float | None = None,
) -> Topology:
    """Dilated loop plus a loop-only rank-1 pilot P = beta * ||A||_2 * u0 v0^T
    (C1): a spare path with its own gain stage, never written into the
    stored weights. u0, v0 default to random-sign flat unit vectors. The pilot is
    split into non-negative P+/P- so it too is realizable. ``loads_node`` adds
    |P| row/col sums to the node load (default False: spare-path, negligible).
    ``s1_ref`` fixes the pilot scale to a given sigma_1 (e.g. the healthy tile's) so a
    faulty copy of the array carries the SAME physical pilot, not one re-scaled to its
    own slightly different sigma_1."""
    gp, gn = _split(g_pos, g_neg)
    m, n = gp.shape
    a = gp - gn
    rng = rng or np.random.default_rng(0)
    if u0 is None:
        u0 = np.sign(rng.standard_normal(m)) / np.sqrt(m)
    if v0 is None:
        v0 = np.sign(rng.standard_normal(n)) / np.sqrt(n)
    u0 = np.asarray(u0, dtype=np.float64) / np.linalg.norm(u0)
    v0 = np.asarray(v0, dtype=np.float64) / np.linalg.norm(v0)
    s1 = float(np.linalg.norm(a, 2)) if s1_ref is None else float(s1_ref)
    p = beta * s1 * np.outer(u0, v0)
    tot = gp + gn
    load_u, load_v = tot.sum(1), tot.sum(0)
    if loads_node:
        load_u = load_u + np.abs(p).sum(1)
        load_v = load_v + np.abs(p).sum(0)
    return Topology(
        kind="pilot",
        mp=_dilate(gp + np.clip(p, 0, None)),
        mn=_dilate(gn + np.clip(-p, 0, None)),
        load=np.concatenate([load_u, load_v]),
        n_u=m,
        n_v=n,
        base_eff=a + p,
        ref_kind="svd_dilation",
        notes={"beta": beta, "u0": u0, "v0": v0, "sigma1_A": s1},
    )


def differential(g_pos, g_neg, g_ref) -> Topology:
    """Dilated loop on E = A_now - A_ref (C2). A second, reference array
    (its own conductances, drift, programming error) is wired with opposite sign
    to the same nodes; both arrays load every node, so S is the sum over all four
    conductance planes. ``g_ref`` is either (G+_ref, G-_ref) or a signed matrix."""
    gp, gn = _split(g_pos, g_neg)
    if isinstance(g_ref, tuple | list):
        rp, rn = _split(g_ref[0], g_ref[1])
    else:
        rp, rn = _split(g_ref, None)
    if rp.shape != gp.shape:
        raise ValueError("reference array shape must match the live array")
    m, n = gp.shape
    mp_cells = gp + rn  # A_now's G+ and the reference's G- both push the same direction
    mn_cells = gn + rp
    tot = gp + gn + rp + rn
    return Topology(
        kind="differential",
        mp=_dilate(mp_cells),
        mn=_dilate(mn_cells),
        load=np.concatenate([tot.sum(1), tot.sum(0)]),
        n_u=m,
        n_v=n,
        base_eff=(gp - gn) - (rp - rn),
        ref_kind="svd_dilation",
    )
