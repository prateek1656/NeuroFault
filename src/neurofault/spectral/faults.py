"""Device-level fault injection with weight-level ground truth.

WHY device level: a stuck memristor pins ONE device (G+ or G-) of a differential
pair, and the resulting weight error depends on the cell's programmed state
(a weight stored on G+ with G- already at g_min barely changes if G- sticks at
g_min, but changes by the full programmed value if G+ sticks at g_min). So
`inject` edits conductances and reports `FaultTruth.delta_w` as the MEASURED
decoded-weight change, never an assumed one.

Semantics per encoding (documented choices, not sourced):
- diff_pair: pin G+ / G- (spec.device; "either" = per-cell coin flip) to g_min
  (hrs) or g_max (lrs).
- offset_refcol: the only variable device is g_pos (the reference column is
  shared hardware and not a per-cell fault); device "neg" is rejected, "either"
  means pos. hrs/lrs pin g_pos to g_min/g_max.
- signed_ideal: no physical rails. hrs -> weight 0 (g=0); lrs -> g = +-g_max_uS
  with a random sign (the old probe's "weight to +-wmax"); device must be pos/either.
Random draws are consumed in a fixed order (cells, rail coin, device coin, sign)
regardless of encoding so the same seed gives the same fault map across encodings.

Geometry reuses faults.injection._select_fault_indices (works on plain shapes).
Clustered selection can return fewer than k cells when blobs overlap or hit the
edge; we top up with random unused cells so k is exact (flagged in tests).
"""

from __future__ import annotations

import torch

from neurofault.faults.injection import _select_fault_indices
from neurofault.spectral.types import ConductanceState, FaultSpec, FaultTruth


def _n_faults(spec: FaultSpec, total: int, what: str) -> int:
    if (spec.k is None) == (spec.density is None):
        raise ValueError("FaultSpec needs exactly one of k or density")
    k = spec.k if spec.k is not None else round(spec.density * total)
    if k < 0 or k > total:
        raise ValueError(f"{what}: k={k} outside [0, {total}]")
    return int(k)


def _select_cells(spec: FaultSpec, m: int, n: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """Flat indices (k,) of faulty cells."""
    dev = gen.device
    dist = {"stuck_random": "random", "stuck_clustered": "clustered", "stuck_gradient": "gradient"}
    flat = _select_fault_indices(m, n, k, dist[spec.kind], dev, gen)
    if flat.numel() < k:  # clustered shortfall (overlapping / edge-clipped blobs)
        used = torch.zeros(m * n, dtype=torch.bool, device=dev)
        used[flat] = True
        free = (~used).nonzero().squeeze(1)
        extra = free[torch.randperm(free.numel(), device=dev, generator=gen)[: k - flat.numel()]]
        flat = torch.cat([flat, extra])
    return flat


def inject(
    state: ConductanceState, spec: FaultSpec, gen: torch.Generator
) -> tuple[ConductanceState, FaultTruth]:
    """Return (faulty copy of state, ground truth). The input state is not modified."""
    m, n = state.shape
    kind = state.enc.kind
    if spec.device == "neg" and kind != "diff_pair":
        raise ValueError(f"device='neg' is meaningless for encoding {kind!r}")
    new = state.clone()
    w_before = state.weights()

    if spec.kind in ("line_row", "line_col"):
        cells, devs = _line_cells(new, spec, gen, m, n)
    else:
        k = _n_faults(spec, m * n, spec.kind)
        flat = _select_cells(spec, m, n, k, gen)
        cells = torch.stack([flat // n, flat % n], dim=1)
        devs = _draw_devices(spec, k, gen, kind == "diff_pair")
        lrs = _draw_rails(spec, k, gen)
        _pin(new, cells, devs, lrs, gen)

    cells = cells.to(state.g_pos.device)
    dw = (new.weights() - w_before)[cells[:, 0], cells[:, 1]]
    if devs is not None:
        devs = devs.to(state.g_pos.device)
    truth = FaultTruth(
        cells=cells,
        delta_w=dw,
        rows=cells[:, 0].unique(),
        cols=cells[:, 1].unique(),
        devices=devs,
    )
    return new, truth


def _draw_devices(spec: FaultSpec, k: int, gen: torch.Generator, pair: bool = True) -> torch.Tensor:
    coin = torch.rand(k, generator=gen, device=gen.device) < 0.5  # always drawn (alignment)
    if spec.device == "pos" or not pair:
        return torch.zeros(k, dtype=torch.long, device=gen.device)
    if spec.device == "neg":
        return torch.ones(k, dtype=torch.long, device=gen.device)
    return coin.long()


def _draw_rails(spec: FaultSpec, k: int, gen: torch.Generator) -> torch.Tensor:
    """bool (k,): True = lrs (g_max)."""
    coin = torch.rand(k, generator=gen, device=gen.device) < 0.5
    if spec.rail == "hrs":
        return torch.zeros(k, dtype=torch.bool, device=gen.device)
    if spec.rail == "lrs":
        return torch.ones(k, dtype=torch.bool, device=gen.device)
    return coin


def _pin(
    st: ConductanceState,
    cells: torch.Tensor,
    devs: torch.Tensor,
    lrs: torch.Tensor,
    gen: torch.Generator,
) -> None:
    """Pin device devs[i] of cell cells[i] at its rail; record in st.stuck (in place)."""
    r, c = cells[:, 0].to(st.g_pos.device), cells[:, 1].to(st.g_pos.device)
    d, lrs = devs.to(st.g_pos.device), lrs.to(st.g_pos.device)
    g_min, g_max = st.enc.g_min_uS, st.enc.g_max_uS
    sign = torch.where(torch.rand(len(r), generator=gen, device=gen.device) < 0.5, -1.0, 1.0)
    sign = sign.to(device=st.g_pos.device, dtype=st.g_pos.dtype)
    if st.enc.kind == "signed_ideal":
        val = torch.where(lrs, sign * g_max, torch.zeros_like(sign))
    else:
        val = torch.where(lrs, torch.full_like(sign, g_max), torch.full_like(sign, g_min))
    pos = d == 0
    st.g_pos[r[pos], c[pos]] = val[pos]
    st.stuck[r[pos], c[pos], 0] = True
    neg = ~pos
    if neg.any():
        st.g_neg[r[neg], c[neg]] = val[neg]
        st.stuck[r[neg], c[neg], 1] = True


def _line_cells(
    st: ConductanceState, spec: FaultSpec, gen: torch.Generator, m: int, n: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Whole-row / whole-column defect. Returns (cells, devices); devices is None
    for line_mode='open' (both devices of a pair are zeroed, so no single device
    is responsible)."""
    by_row = spec.kind == "line_row"
    n_lines_total = m if by_row else n
    k = _n_faults(spec, n_lines_total, spec.kind)
    dev = gen.device
    lines = torch.randperm(n_lines_total, device=dev, generator=gen)[:k]
    span = torch.arange(n if by_row else m, device=dev)
    line_g = lines.repeat_interleave(len(span))
    span_g = span.repeat(k)
    cells = torch.stack([line_g, span_g] if by_row else [span_g, line_g], dim=1)
    # one device / rail decision per LINE (a word/bit-line defect is a shared wire)
    devs_l = _draw_devices(spec, k, gen, st.enc.kind == "diff_pair")
    lrs_l = _draw_rails(spec, k, gen)
    if spec.line_mode == "open":
        _open(st, cells)
        return cells, None
    reps = len(span)
    devs = devs_l.repeat_interleave(reps)
    lrs = lrs_l.repeat_interleave(reps)
    _pin(st, cells, devs, lrs, gen)
    return cells, devs


def _open(st: ConductanceState, cells: torch.Tensor) -> None:
    """Open line: decoded weight 0 and every device on the line is dead (stuck)."""
    r, c = cells[:, 0].to(st.g_pos.device), cells[:, 1].to(st.g_pos.device)
    if st.enc.kind == "diff_pair":
        st.g_pos[r, c] = st.enc.g_min_uS
        st.g_neg[r, c] = st.enc.g_min_uS
        st.stuck[r, c, :] = True
    elif st.enc.kind == "offset_refcol":
        st.g_pos[r, c] = st.g_ref()  # decodes to weight 0
        st.stuck[r, c, 0] = True
    else:
        st.g_pos[r, c] = 0.0
        st.stuck[r, c, 0] = True
