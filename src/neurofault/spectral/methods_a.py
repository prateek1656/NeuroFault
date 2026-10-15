"""Direction-A compensators: M4 field fit, M3 input-subspace, M2 sketch.

WHY M4: patterned drift is multiplicative, W(t) ~= W0 o (1 + F) with F a smooth field, so
E = W0 o F is NOT low-rank even when F is. M4 therefore estimates a low-dimensional F from a
few forward (and backward) probe reads against the stored t0 reads:
    forward  : (W0 o F) X  = Y(t) - Y(t0)         (m, Pf)
    backward : (W0 o F)^T Z = Z-read(t) - Z-read(t0) (n, Pb)
Both are LINEAR in F. Two parametrisations of F:
  - "cos"  : F = sum_b c_b phi_b(i, j), phi = low-order 2D cosines over (row, col); linear LS
             (ridge) in c.
  - "rank" : F = sum_k u_k v_k^T (rank r), i.e. W0 o F = sum_k diag(u_k) W0 diag(v_k);
             bilinear, solved by ALS (each half-step is a linear LS in U or V).
The correction removes the estimated drift error digitally: W_eff = W(t) - W0 o F_hat. W(t)
(Bindable.now) is used ONLY to materialise W_eff, never for estimation; W0 is the stored digital
reference (the programming target) and enters only the known linear design.

M3 (input-subspace / Galerkin): probe with the top-k input PCs U_k, correct
W_eff = W(t) + (Y0 - Y') U_k^T. M2 (ResidualSketch): two-sided single-view sketch of the
residual of an already gain-corrected operator, rank-k truncated.
"""

from __future__ import annotations

import math

import torch

from neurofault.spectral.baselines.base import (
    Bindable,
    Correction,
    ReadCost,
    TileReaderLike,
    make_probes,
)
from neurofault.spectral.types import DTYPE, make_generator


def cosine_basis(m: int, n: int, order: int) -> torch.Tensor:
    """(B, m, n) separable 2D cosines cos(pi a (i+.5)/m) cos(pi b (j+.5)/n), a, b < order."""
    i = (torch.arange(m, dtype=DTYPE) + 0.5) / m
    j = (torch.arange(n, dtype=DTYPE) + 0.5) / n
    ca = torch.cos(math.pi * torch.arange(order, dtype=DTYPE)[:, None] * i[None, :])  # (a, m)
    cb = torch.cos(math.pi * torch.arange(order, dtype=DTYPE)[:, None] * j[None, :])  # (b, n)
    return torch.einsum("ai,bj->abij", ca, cb).reshape(order * order, m, n)


def _ridge_solve(a: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    """argmin ||a x - y||^2 + lam ||x||^2 with lam = ridge * mean diag(a^T a)."""
    g = a.T @ a
    lam = ridge * float(torch.diagonal(g).mean()) + 1e-300
    return torch.linalg.solve(g + lam * torch.eye(g.shape[0], dtype=DTYPE), a.T @ y)


def _solve_rows(
    w: torch.Tensor,
    pf: torch.Tensor,
    df: torch.Tensor,
    pb: torch.Tensor,
    db: torch.Tensor,
    other: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    """Solve F = U other^T for U (m, r) given W (m, n), forward obs (W o F) pf = df (m, Pf) and
    backward obs (W o F)^T pb = db (n, Pb); `other` is (n, r)."""
    m, n = w.shape
    r = other.shape[1]
    # forward block is block-diagonal in rows: S[k][i, p] = sum_j W_ij v_jk pf_jp
    s = torch.einsum("ij,jk,jp->ikp", w, other, pf)  # (m, r, Pf)
    jf = torch.zeros(m, pf.shape[1], m, r, dtype=DTYPE)
    idx = torch.arange(m)
    jf[idx, :, idx, :] = s.permute(0, 2, 1)
    blocks = [jf.reshape(m * pf.shape[1], m * r)]
    rhs = [df.reshape(-1)]
    if pb is not None and pb.shape[1] > 0:
        jb = torch.einsum("jk,ij,iq->jqik", other, w, pb)  # (n, Pb, m, r)
        blocks.append(jb.reshape(n * pb.shape[1], m * r))
        rhs.append(db.reshape(-1))
    a = torch.cat(blocks, 0)
    y = torch.cat(rhs, 0)
    return _ridge_solve(a, y, ridge).reshape(m, r)


def fit_rank_field(
    w0: torch.Tensor,
    xf: torch.Tensor,
    df: torch.Tensor,
    zb: torch.Tensor | None,
    db: torch.Tensor | None,
    rank: int,
    iters: int = 12,
    ridge: float = 1e-10,
    seed: int = 0,
) -> torch.Tensor:
    """ALS for F = U V^T (rank r) from linear observations of W0 o F. Returns F (m, n)."""
    m, n = w0.shape
    gen = make_generator(seed, "field_fit_init")
    v = torch.randn(n, rank, generator=gen, dtype=DTYPE) * 0.1
    v[:, 0] = 1.0
    wt = w0.T.contiguous()
    pbt = xf  # backward role of the transposed problem
    u = torch.zeros(m, rank, dtype=DTYPE)
    for _ in range(iters):
        u = _solve_rows(w0, xf, df, zb, db, v, ridge)
        # V-step on the transposed problem: (W^T o F^T) acts forward on Z, backward on X
        if zb is not None:
            v = _solve_rows(wt, zb, db, pbt, df, u, ridge)
        else:  # forward-only: only the backward-role term of the transposed problem remains
            empty = torch.zeros(m, 0, dtype=DTYPE)
            v = _solve_rows(wt, empty, torch.zeros(n, 0, dtype=DTYPE), pbt, df, u, ridge)
    return u @ v.T


def fit_basis_field(
    w0: torch.Tensor,
    basis: torch.Tensor,
    xf: torch.Tensor,
    df: torch.Tensor,
    zb: torch.Tensor | None,
    db: torch.Tensor | None,
    ridge: float = 1e-8,
) -> torch.Tensor:
    """Linear (ridge) LS for F = sum_b c_b basis_b from the same linear observations."""
    c = fit_basis_coeffs(w0, basis, xf, df, zb, db, ridge)
    return torch.einsum("b,bij->ij", c, basis)


def fit_basis_coeffs(
    w0: torch.Tensor,
    basis: torch.Tensor,
    xf: torch.Tensor,
    df: torch.Tensor,
    zb: torch.Tensor | None,
    db: torch.Tensor | None,
    ridge: float = 1e-8,
) -> torch.Tensor:
    """The (B,) coefficient vector c of fit_basis_field (c.reshape(order, order) is the 4x4
    coefficient matrix C of F = Ca^T C Cb for the separable cosine basis)."""
    wb = w0[None] * basis  # (B, m, n)
    cols = [torch.einsum("bij,jp->bip", wb, xf).reshape(basis.shape[0], -1)]  # (B, m*Pf)
    rhs = [df.reshape(-1)]
    if zb is not None:
        cb = torch.einsum("bij,iq->bjq", wb, zb).reshape(basis.shape[0], -1)
        cols.append(cb)
        rhs.append(db.reshape(-1))
    a = torch.cat(cols, 1).T  # (N, B)
    return _ridge_solve(a, torch.cat(rhs), ridge)


# ---- deployable (zero / few extra inference passes) form of the field correction -----------
#
# The on-chip array holds W1 = W(t) = W0 o (1 + F). The exact correction is the elementwise
# multiplier G = 1 / (1 + F_hat): W1 o G = W0. G is smooth, so it is projected onto the same
# separable cosine span (order x order coefficient matrix C_G, G ~= Ca^T C_G Cb), and C_G is
# truncated to rank r by SVD: C_G ~= sum_k s_k u_k v_k^T gives
#     W1 o G_r = sum_k diag(a_k) W1 diag(b_k),   a_k = sqrt(s_k) Ca^T u_k,  b_k = sqrt(s_k) Cb^T v_k.
# Each term is ONE analog MVM on the array with input scaling b_k and output scaling a_k (digital
# periphery). A rank-1 correction is therefore a single pass = the base pass (0 extra); rank r
# costs r - 1 extra analog passes. (The first-order form W1 - sum_k diag(u_k) W1 diag(v_k) used
# in the H5 registration note needs one more pass than G-form at equal rank and is only accurate
# for small F; G-form is exact for a correct F_hat.)


def _cos_factors(m: int, n: int, order: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Ca (order, m), Cb (order, n): the separable cosine factors of cosine_basis."""
    i = (torch.arange(m, dtype=DTYPE) + 0.5) / m
    j = (torch.arange(n, dtype=DTYPE) + 0.5) / n
    k = torch.arange(order, dtype=DTYPE)[:, None]
    return torch.cos(math.pi * k * i[None, :]), torch.cos(math.pi * k * j[None, :])


def project_cosine_coeffs(g: torch.Tensor, order: int) -> torch.Tensor:
    """LS coefficient matrix C (order, order) of g (m, n) in the separable cosine span."""
    ca, cb = _cos_factors(*g.shape, order)
    left = torch.linalg.solve(ca @ ca.T, ca)  # (order, m)
    right = torch.linalg.solve(cb @ cb.T, cb)  # (order, n)
    return left @ g @ right.T


def rank_factors(c: torch.Tensor, shape: tuple[int, int], rank: int, tol: float = 1e-9):
    """Top-`rank` SVD terms of coefficient matrix c as per-row / per-column vectors:
    returns (A (m, k), Bv (n, k)) with sum_k A[:, k] Bv[:, k]^T the truncated field; k <= rank
    counts only singular values above tol * s_max."""
    order = c.shape[0]
    ca, cb = _cos_factors(*shape, order)
    u, sv, vh = torch.linalg.svd(c)
    keep = int(min(rank, int((sv > tol * sv[0]).sum()))) if float(sv[0]) > 0 else 1
    keep = max(keep, 1)
    root = sv[:keep].sqrt()
    return (ca.T @ u[:, :keep]) * root, (cb.T @ vh[:keep].T) * root


def deployable_correction(
    wt: torch.Tensor, f_hat: torch.Tensor, order: int, rank: int | None, floor: float = 0.1
) -> tuple[torch.Tensor, int]:
    """(W_eff, n_terms): W_eff = W1 o G_r with G = 1/max(1 + F_hat, floor) projected on the
    order x order cosine span and truncated to `rank` (None = full, no truncation);
    n_terms = analog passes the correction needs (extra passes = n_terms - 1)."""
    g = 1.0 / (1.0 + f_hat).clamp_min(floor)
    c = project_cosine_coeffs(g, order)
    a, b = rank_factors(c, tuple(g.shape), order * order if rank is None else rank)
    return wt * (a @ b.T), a.shape[1]


class FieldFit(Bindable):
    """M4 field-fit compensator. model: "cos" (smooth basis) | "rank" (rank-r ALS)."""

    def __init__(
        self,
        model: str = "cos",
        order: int = 4,
        rank: int = 1,
        p_fwd: int = 8,
        p_bwd: int = 8,
        iters: int = 12,
        ridge: float | None = None,
        seed: int = 0,
        deploy: bool = False,
        deploy_rank: int | None = None,
    ):
        """deploy=True returns the deployable G-form W_eff = W(t) o G_r (see deployable_correction;
        deploy_rank None = full, r=1 needs no extra inference pass; meta['extra_passes'] = r_eff-1)
        instead of the digital-reference form W(t) - W0 o F_hat. model='cos' only."""
        if deploy and model != "cos":
            raise ValueError("deploy requires model='cos'")
        self.deploy, self.deploy_rank = deploy, deploy_rank
        if model not in ("cos", "rank"):
            raise ValueError("model must be cos|rank")
        self.model, self.order, self.rank = model, order, rank
        self.p_fwd, self.p_bwd, self.iters = p_fwd, p_bwd, iters
        self.ridge = (1e-8 if model == "cos" else 1e-10) if ridge is None else ridge
        self.gen = make_generator(seed, f"field_fit_{model}")
        self.seed = seed
        par = order if model == "cos" else rank
        self.name = f"M4_fieldfit_{model}{par}_f{p_fwd}b{p_bwd}"
        self.xf = self.zb = self.y0 = self.z0 = None
        self._basis: torch.Tensor | None = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        m, n = reader.shape
        self.xf = make_probes("rademacher", n, self.p_fwd, self.gen)
        self.y0 = reader.mvm(self.xf)
        self.zb = self.z0 = None
        if reader.backward_supported and self.p_bwd > 0:
            self.zb = make_probes("rademacher", m, self.p_bwd, self.gen)
            self.z0 = reader.mvm_t(self.zb)
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.xf is not None, "calibrate() first"
        start = reader.cost.copy()
        df = reader.mvm(self.xf) - self.y0
        db = (reader.mvm_t(self.zb) - self.z0) if self.zb is not None else None
        w0 = self.ref()
        if self.model == "cos":
            if self._basis is None:
                self._basis = cosine_basis(*w0.shape, self.order)
            f = fit_basis_field(w0, self._basis, self.xf, df, self.zb, db, self.ridge)
        else:
            f = fit_rank_field(w0, self.xf, df, self.zb, db, self.rank, self.iters, self.ridge)
        if self.deploy:
            w_eff, terms = deployable_correction(self.now(), f, self.order, self.deploy_rank)
            return Correction(w_eff, reader.cost - start, {"F": f, "extra_passes": terms - 1})
        e_hat = w0 * f
        return Correction(self.now() - e_hat, reader.cost - start, {"F": f, "E_hat": e_hat})


class InputSubspace(Bindable):
    """M3: probe with the top-k input PCs U_k (n, k); W_eff = W(t) + (Y0 - Y') U_k^T."""

    def __init__(self, u_k: torch.Tensor):
        self.u = u_k.to(DTYPE)
        self.name = f"M3_input_subspace_k{self.u.shape[1]}"
        self.y0: torch.Tensor | None = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        self.y0 = reader.mvm(self.u)
        return reader.cost - start

    def update(self, reader: TileReaderLike, t: float) -> Correction:
        assert self.y0 is not None, "calibrate() first"
        start = reader.cost.copy()
        d = self.y0 - reader.mvm(self.u)
        return Correction(self.now() + d @ self.u.T, reader.cost - start, {})


class ResidualSketch:
    """M2: two-sided single-view sketch of the residual of a gain-corrected operator.

    The residual is R = diag(g) alpha W(t) - W_ref where the references Y0/Z0 are the t0 reads.
    update(reader, t, row_gain) sketches R from g o read(Omega) - Y0 and the backward analogue,
    and returns the rank-k estimate R_k (m, n); the caller subtracts it from its operator."""

    def __init__(self, k: int = 8, s: int = 4, seed: int = 0):
        self.k, self.s = k, s
        self.gen = make_generator(seed, f"resid_sketch_{k}")
        self.name = f"M2_sketch_k{k}_s{s}"
        self.om = self.ps = self.y0 = self.z0 = None

    def calibrate(self, reader: TileReaderLike) -> ReadCost:
        start = reader.cost.copy()
        m, n = reader.shape
        p = self.k + self.s
        self.om = make_probes("rademacher", n, p, self.gen)
        self.ps = make_probes("rademacher", m, p, self.gen)
        self.y0 = reader.mvm(self.om)
        self.z0 = reader.mvm_t(self.ps)
        return reader.cost - start

    def update(
        self,
        reader: TileReaderLike,
        t: float,
        row_gain: torch.Tensor | None = None,
        col_gain: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ReadCost]:
        """Sketch R = diag(g) W(t) diag(h) - W_ref through the analog reads (g, h digital)."""
        start = reader.cost.copy()
        m, n = reader.shape
        g = torch.ones(m, dtype=DTYPE) if row_gain is None else row_gain
        h = torch.ones(n, dtype=DTYPE) if col_gain is None else col_gain
        y = g[:, None] * reader.mvm(h[:, None] * self.om) - self.y0  # R Omega
        z = h[:, None] * reader.mvm_t(g[:, None] * self.ps) - self.z0  # R^T Psi
        core = self.ps.T @ y
        uc, sc, vhc = torch.linalg.svd(core)
        kc = min(self.k, sc.numel())
        cinv = (vhc[:kc].T / sc[:kc].clamp(min=1e-300)) @ uc[:, :kc].T
        qa, ra = torch.linalg.qr(y @ cinv)
        qb, rb = torch.linalg.qr(z)
        u, sv, vh = torch.linalg.svd(ra @ rb.T)
        k = min(self.k, sv.numel())
        r_k = ((qa @ u[:, :k]) * sv[:k]) @ (qb @ vh[:k].T).T
        return r_k, reader.cost - start
