"""Shared contract for the spectral-diagnosis vertical.

Every module under neurofault.spectral codes against these types. They are
written first, so parallel builders agree on one
interface - change them only with a recorded reason.

Conventions:
- A tile weight matrix W has shape (m, n) = (outputs, inputs), a block of
  state_dict[name].flatten(1) - the same orientation as CrossSimAccessor.read().
  Forward MVM y = W x (x in R^n),
  backward MVM W^T y (y in R^m).
- float64 by default: sigma_1/sigma_2 is ~1.02-1.11 on trained tiles, so SVD
  and eigenvector work needs the precision. float32 is opt-in per call site.
- Time t is seconds since programming completed; t0 = 20 s (AIHWKit) by default.
- Conductances are in microsiemens (uS). Signed weights are recovered by
  ConductanceState.weights().
- All randomness goes through an explicit torch.Generator from make_generator(),
  one independent stream per purpose (tiles, faults, drift, read noise, ...), so
  arms that share a seed see identical faults and drift regardless of how much
  randomness other parts of the pipeline consumed.
- Faults are specified per DEVICE (stuck[m, n, 2]: index 0 = G+, 1 = G-) and
  reported per WEIGHT (FaultTruth.delta_w): with differential pairs the weight
  change of a stuck device depends on the cell's programmed state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields, replace
from typing import Literal, Protocol, runtime_checkable

import torch

DTYPE = torch.float64
T0_S = 20.0


def make_generator(seed: int, stream: str, device: str | torch.device = "cpu") -> torch.Generator:
    """Independent, replayable RNG stream derived from (seed, stream name)."""
    digest = hashlib.sha256(f"{seed}:{stream}".encode()).digest()
    gen = torch.Generator(device=device)
    gen.manual_seed(int.from_bytes(digest[:8], "little") & ((1 << 63) - 1))
    return gen


# --------------------------------------------------------------------------- tiles


@dataclass(frozen=True)
class TileSpec:
    """Where a tile comes from: rows r0:r0+m, cols c0:c0+n of layer.flatten(1).

    ckpt_seed = -1 marks a synthetic tile (no checkpoint)."""

    ckpt_seed: int
    layer: str
    r0: int
    c0: int
    m: int
    n: int
    tile_size: int


@dataclass
class Tile:
    spec: TileSpec
    w: torch.Tensor  # (m, n) float64, signed, checkpoint weight units


# --------------------------------------------------------------------------- encoding


@dataclass(frozen=True)
class EncodingSpec:
    kind: Literal["signed_ideal", "diff_pair", "offset_refcol"] = "diff_pair"
    g_max_uS: float = 25.0  # AIHWKit single-pair converter default [sourced]
    g_min_frac: float = 0.0  # g_min / g_max: 0 (AIHWKit) | 0.01 (repo) | 0.05 (assumed)
    scale_policy: Literal["tile_absmax", "layer_absmax"] = "tile_absmax"

    @property
    def g_min_uS(self) -> float:
        return self.g_min_frac * self.g_max_uS


@dataclass
class ConductanceState:
    """Physical state of one tile at time t.

    diff_pair:     w = (g_pos - g_neg) / scale, both in [g_min, g_max].
    offset_refcol: w = (g_pos - g_ref) / scale with a fixed reference level
                   g_ref = (g_min + g_max) / 2; g_neg is None.
    signed_ideal:  g_pos holds signed values directly (no physical rails);
                   g_neg is None; w = g_pos / scale.
    """

    g_pos: torch.Tensor  # (m, n) uS
    g_neg: torch.Tensor | None  # (m, n) uS, None unless diff_pair
    scale: float  # uS per weight unit
    enc: EncodingSpec
    t: float = 0.0
    stuck: torch.Tensor | None = None  # (m, n, 2) bool; device pinned (excluded from drift)

    def __post_init__(self) -> None:
        if self.stuck is None:
            m, n = self.g_pos.shape
            self.stuck = torch.zeros(m, n, 2, dtype=torch.bool, device=self.g_pos.device)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.g_pos.shape)

    def g_ref(self) -> float:
        return 0.5 * (self.enc.g_min_uS + self.enc.g_max_uS)

    def weights(self) -> torch.Tensor:
        """Decode to signed effective weights (m, n)."""
        if self.enc.kind == "diff_pair":
            assert self.g_neg is not None
            return (self.g_pos - self.g_neg) / self.scale
        if self.enc.kind == "offset_refcol":
            return (self.g_pos - self.g_ref()) / self.scale
        return self.g_pos / self.scale

    def total_conductance(self) -> torch.Tensor:
        """g_pos + g_neg per cell (row sums give S_i for the circuit model)."""
        if self.g_neg is None:
            return self.g_pos.abs()
        return self.g_pos + self.g_neg

    def clone(self, **changes) -> ConductanceState:
        base = replace(
            self,
            g_pos=self.g_pos.clone(),
            g_neg=None if self.g_neg is None else self.g_neg.clone(),
            stuck=self.stuck.clone(),
        )
        return replace(base, **changes) if changes else base


# --------------------------------------------------------------------------- noise


@dataclass(frozen=True)
class ProgNoise:
    kind: Literal["none", "aihwkit_pcm_poly", "gauss_rel"] = "none"
    rel_sigma: float = 0.0  # for gauss_rel: sigma as a fraction of g_max


@dataclass(frozen=True)
class ReadNoise:
    mode: Literal["per_cell", "output_referred", "both"] = "per_cell"
    sigma_cell_frac: float = 0.02  # of tile max |w|; 0.02 = the first probe's value [assumed]
    sigma_out_frac: float = 0.0  # of rms(y) of the probe family (TIA/ADC referred) [assumed]
    one_over_f: bool = False  # AIHWKit PCM 1/f read noise [sourced]
    persistence: Literal["fresh", "frozen"] = "fresh"
    adc_bits: int | None = 8  # repo defaults (CrossbarConfig)
    dac_bits: int | None = 8


# --------------------------------------------------------------------------- cost


@dataclass
class ReadCost:
    """Counted cost of a procedure. TileReader accumulates into one of these;
    baselines/detectors return the cost of each call. `images` counts
    calibration images (AdaBS)."""

    mvm_fwd: int = 0
    mvm_bwd: int = 0
    adc: int = 0
    dac: int = 0
    cell_reads: int = 0
    writes: int = 0
    images: int = 0

    def __add__(self, other: ReadCost) -> ReadCost:
        return ReadCost(
            **{f.name: getattr(self, f.name) + getattr(other, f.name) for f in fields(self)}
        )

    def __sub__(self, other: ReadCost) -> ReadCost:
        return ReadCost(
            **{f.name: getattr(self, f.name) - getattr(other, f.name) for f in fields(self)}
        )

    def copy(self) -> ReadCost:
        return replace(self)

    def as_dict(self) -> dict[str, int]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------- faults


@dataclass(frozen=True)
class FaultSpec:
    kind: Literal["stuck_random", "stuck_clustered", "stuck_gradient", "line_row", "line_col"]
    k: int | None = None  # number of faulty cells (or lines for line_*)
    density: float | None = None  # alternative to k: fraction of cells
    rail: Literal["hrs", "lrs", "mixed"] = "mixed"  # hrs = g_min, lrs = g_max
    device: Literal["pos", "neg", "either"] = "either"  # which device of the pair is pinned
    line_mode: Literal["stuck_rail", "open"] = "stuck_rail"


@dataclass
class FaultTruth:
    cells: torch.Tensor  # (K, 2) long: (row, col) of weight cells affected
    delta_w: torch.Tensor  # (K,) change in decoded weight (state dependent for pairs)
    rows: torch.Tensor  # unique affected rows
    cols: torch.Tensor  # unique affected cols
    devices: torch.Tensor | None = None  # (K,) long: 0 = G+, 1 = G-


@dataclass
class FaultEstimate:
    """A localizer's output. score is optional per-cell confidence."""

    cells: torch.Tensor  # (K_hat, 2) long
    rows: torch.Tensor | None = None
    cols: torch.Tensor | None = None
    score: torch.Tensor | None = None  # (K_hat,)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- params


@dataclass(frozen=True)
class ParamRecord:
    value: object
    status: Literal["sourced", "assumed"]
    source: str
    verified: bool
    sweep: tuple | None = None


# --------------------------------------------------------------------------- drift


@runtime_checkable
class DriftRealization(Protocol):
    """All per-device randomness is drawn once in DriftModel.sample(); at(t) is
    deterministic given that draw (a device keeps its own nu at every t)."""

    def at(self, t: float) -> ConductanceState: ...

    def sweep(self, ts: list[float]) -> list[ConductanceState]: ...


@runtime_checkable
class DriftModel(Protocol):
    name: str

    def sample(self, state0: ConductanceState, gen: torch.Generator) -> DriftRealization: ...


# --------------------------------------------------------------------------- readers and methods


@runtime_checkable
class TileReaderLike(Protocol):
    """What baselines/detectors/localizers may touch. They never see the
    matrix directly - only counted, noisy reads (estimators must not be able to cheat by reading the full matrix for free)."""

    cost: ReadCost
    backward_supported: bool

    @property
    def shape(self) -> tuple[int, int]: ...

    def mvm(self, x: torch.Tensor) -> torch.Tensor: ...  # (n, p) -> (m, p)

    def mvm_t(self, y: torch.Tensor) -> torch.Tensor: ...  # (m, p) -> (n, p)

    def full_readout(self) -> torch.Tensor: ...  # (m, n), costs m*n cell reads


@dataclass
class Correction:
    W_eff: torch.Tensor  # corrected effective signed weights (m, n)
    cost: ReadCost
    meta: dict = field(default_factory=dict)


@runtime_checkable
class Compensator(Protocol):
    name: str

    def calibrate(self, reader: TileReaderLike) -> ReadCost: ...  # right after programming

    def update(self, reader: TileReaderLike, t: float) -> Correction: ...


@runtime_checkable
class Detector(Protocol):
    name: str

    def calibrate(self, reader: TileReaderLike) -> ReadCost: ...

    def statistic(self, reader: TileReaderLike, t: float) -> tuple[float, ReadCost]: ...


@runtime_checkable
class Localizer(Protocol):
    name: str

    def calibrate(self, reader: TileReaderLike) -> ReadCost: ...

    def localize(self, reader: TileReaderLike, t: float) -> tuple[FaultEstimate, ReadCost]: ...
