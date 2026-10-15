"""Exponential recovery model, online patience stopper, and an OFFLINE
lifetime simulator used to grid-search the co-mitigation controller's
parameters.

Everything here is a calibration tool, not a claim about hardware: the
simulator is only as good as the fitted curves fed into it, and the real
controller is always evaluated by actually running it. Pure numpy/python
(scipy is imported lazily, only inside `fit_exponential_recovery`); no torch.

Modeling assumptions of `simulate_lifetime` (each one a simplification):
  1. Recovery after a wave follows acc(t) = A_inf(u) - (A_inf(u) - A_post(u)) *
     exp(-t/tau_eta) where u is the cumulative UNREMAPPED stuck fraction,
     tau_eta is the fitted time constant for the burst's learning rate eta,
     and A_inf(u) / A_post(u) are linearly interpolated from fitted
     (stuck-fraction, level) points (clamped at the ends).
  2. A_post(u) (accuracy immediately after a wave, before any burst) and
     A_inf(u) are both functions of u only; tau does not depend on u.
  3. Remap reduces the EFFECTIVE new-fault fraction: a wave of f new stuck
     cells with r remapped leaves max(f - eff*r, 0) unremapped, where `eff`
     (remap_efficiency, default 1) is how many fault-equivalents one spare
     removes (>1 models saliency-ordered remap beating random). Capacity is in
     the same units as f (fraction of cells).
  4. Pacing: "greedy" spends min(f, remaining) per wave; "paced" caps the
     spend at remaining / (expected remaining waves incl. this one), i.e.
     rho_k = 1/remaining_waves.
  5. Accuracy does not change between waves except through bursts; a burst is
     truncated if the next wave arrives first. Burst batches count the batches
     actually run.
  6. The canary the stopper sees is the true accuracy plus Gaussian noise with
     SE = 100*sqrt(p(1-p)/n_canary) points (binomial sampling of a fixed-size
     canary set, treated as independent between evals, which is optimistic:
     a fixed canary set gives correlated errors). Lifetime accuracy is the
     TRUE (noiseless) accuracy averaged over batches.
  7. The canary gate (drop <= delta skips the burst) is applied to the true
     post-wave drop plus one noise draw.
  8. Bursts do not alter later waves' A_post/A_inf beyond what u captures
     (e.g. no "learned robustness" carry-over).
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Exponential recovery fit
# --------------------------------------------------------------------------


@dataclass
class RecoveryFit:
    """acc(t) = a_inf - (a_inf - a0) * exp(-t/tau); t in batches since the burst
    (or wave) start, accuracies in percent. `converged=False` marks the
    fallback used when the series was too short/flat/noisy to fit."""

    a_inf: float
    a0: float
    tau: float
    rmse: float
    converged: bool = True

    def predict(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return self.a_inf - (self.a_inf - self.a0) * np.exp(-t / self.tau)


def _exp_model(t, a_inf, a0, tau):
    return a_inf - (a_inf - a0) * np.exp(-t / tau)


def fit_exponential_recovery(t, acc) -> RecoveryFit:
    """Fit the exponential recovery model to (t, acc). Never raises: series with
    fewer than 4 points, non-finite values, or an optimizer failure return a
    fallback (a_inf = mean of the last 3 points, a0 = first point, tau = span/3)
    with `converged=False`."""
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(acc, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if t.size:
        order = np.argsort(t)
        t, y = t[order], y[order]
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    span = max(span, 1.0)

    def fallback() -> RecoveryFit:
        if y.size == 0:
            return RecoveryFit(
                a_inf=0.0, a0=0.0, tau=span / 3.0, rmse=float("nan"), converged=False
            )
        fb = RecoveryFit(
            a_inf=float(np.mean(y[-3:])), a0=float(y[0]), tau=span / 3.0, rmse=0.0, converged=False
        )
        fb.rmse = float(np.sqrt(np.mean((fb.predict(t - t[0]) - y) ** 2)))
        return fb

    if y.size < 4:
        return fallback()
    try:
        from scipy.optimize import curve_fit

        t0 = t - t[0]
        p0 = [min(float(y.max()) + 1.0, 100.0), float(y[0]), span / 3.0]
        popt, _ = curve_fit(
            _exp_model,
            t0,
            y,
            p0=p0,
            bounds=([0.0, 0.0, 1.0], [100.0, 100.0, 1e6]),
            maxfev=20000,
        )
    except Exception as e:  # noqa: BLE001 - a bad series must degrade, not raise
        logger.warning("exponential recovery fit failed (%s); using fallback", e)
        return fallback()
    a_inf, a0, tau = (float(x) for x in popt)
    rmse = float(np.sqrt(np.mean((_exp_model(t0, a_inf, a0, tau) - y) ** 2)))
    # tau pinned at a bound means the data carry no information on it
    converged = bool(1.0 < tau < 1e6 * 0.999)
    return RecoveryFit(a_inf=a_inf, a0=a0, tau=tau, rmse=rmse, converged=converged)


# --------------------------------------------------------------------------
# Online patience stopper
# --------------------------------------------------------------------------


class PatienceStopper:
    """Online stop rule for a retraining burst, fed one canary accuracy per eval.

    Evals occur every `eval_every` (m) batches. Stops when the smoothed gain
    falls below `min_gain` (kappa), in accuracy points per eval:

        gain = (mean(last k evals) - mean(previous k evals)) / k

    (two adjacent k-windows, whose centers are k evals apart), evaluated once
    2k observations exist; or when `t_max` batches have elapsed. Averaging
    windows, not endpoints, keeps ~1% canary SE from triggering stops by noise.
    `reason` is None while running, then "plateau" or "t_max".
    """

    def __init__(self, eval_every: int, window: int, min_gain: float, t_max: int):
        if eval_every < 1 or window < 1 or t_max < 1:
            raise ValueError("eval_every, window and t_max must be >= 1")
        self.eval_every = int(eval_every)
        self.window = int(window)
        self.min_gain = float(min_gain)
        self.t_max = int(t_max)
        self.obs: list[float] = []
        self.reason: str | None = None

    @property
    def batches(self) -> int:
        return len(self.obs) * self.eval_every

    @property
    def stopped(self) -> bool:
        return self.reason is not None

    def update(self, canary_acc: float) -> bool:
        """Feed one canary reading; returns True if the burst should stop now."""
        if self.stopped:
            return True
        self.obs.append(float(canary_acc))
        k = self.window
        if len(self.obs) >= 2 * k:
            recent = float(np.mean(self.obs[-k:]))
            prev = float(np.mean(self.obs[-2 * k : -k]))
            if (recent - prev) / k < self.min_gain:
                self.reason = "plateau"
                return True
        if self.batches >= self.t_max:
            self.reason = "t_max"
            return True
        return False


# --------------------------------------------------------------------------
# Offline lifetime simulator
# --------------------------------------------------------------------------


@dataclass
class RecoveryModel:
    """Per-eta recovery description for the simulator. `stuck_fractions`
    (ascending) with matching `a_inf` and `a_post` give the plateau and the
    just-after-wave accuracy as functions of cumulative unremapped stuck
    fraction; tau is the exponential time constant (batches) for this eta."""

    eta: float
    tau: float
    stuck_fractions: list[float]
    a_inf: list[float]
    a_post: list[float]

    def plateau(self, u: float) -> float:
        return float(np.interp(u, self.stuck_fractions, self.a_inf))

    def post_wave(self, u: float) -> float:
        return float(np.interp(u, self.stuck_fractions, self.a_post))


@dataclass
class PolicyParams:
    eta: float
    delta: float  # canary-drop gate: skip burst if drop <= delta (points)
    kappa: float  # patience: min smoothed gain, points per eval
    k: int  # patience window (evals)
    m: int  # batches per eval
    t_max: int  # burst cap (batches)
    pacing: str = "greedy"  # "greedy" | "paced"


@dataclass
class Wave:
    batch: int  # when the wave hits
    new_fault_fraction: float  # newly stuck cells as a fraction of all cells


@dataclass
class LifetimeResult:
    lifetime_mean_acc: float
    final_acc: float
    worst_dip: float
    burst_batches: float
    spares_used: float
    bursts_run: float = 0.0
    stop_reasons: dict = field(default_factory=dict)


def _canary_se(acc: float, n_canary: int) -> float:
    p = min(max(acc / 100.0, 0.0), 1.0)
    return 100.0 * math.sqrt(p * (1.0 - p) / max(n_canary, 1))


def _simulate_once(
    policy: PolicyParams,
    model: RecoveryModel,
    waves: Sequence[Wave],
    horizon: int,
    clean_acc: float,
    remap_capacity: float,
    remap_efficiency: float,
    n_canary: int,
    noise: bool,
    rng: np.random.Generator,
) -> LifetimeResult:
    waves = sorted(waves, key=lambda w: w.batch)
    acc = clean_acc
    u = 0.0
    spares_left = remap_capacity
    area = 0.0  # integral of true acc over batches
    clock = 0
    worst = clean_acc
    burst_total = 0
    bursts = 0
    reasons: dict[str, int] = {}

    for i, wave in enumerate(waves):
        area += acc * (wave.batch - clock)
        clock = wave.batch
        # remap
        remaining_waves = len(waves) - i
        rho = 1.0 if policy.pacing == "greedy" else 1.0 / remaining_waves
        spend = min(wave.new_fault_fraction / max(remap_efficiency, 1e-12), spares_left * rho)
        spares_left -= spend
        u += max(wave.new_fault_fraction - remap_efficiency * spend, 0.0)
        a_post = min(model.post_wave(u), acc)
        a_inf = max(model.plateau(u), a_post)
        next_batch = waves[i + 1].batch if i + 1 < len(waves) else horizon
        window = next_batch - clock

        drop = acc - a_post
        seen_drop = drop + (rng.normal(0.0, _canary_se(acc, n_canary)) if noise else 0.0)
        acc = a_post
        worst = min(worst, acc)
        if seen_drop <= policy.delta or window <= 0:
            area += acc * window
            clock = next_batch
            continue

        stopper = PatienceStopper(policy.m, policy.k, policy.kappa, policy.t_max)
        ran = 0
        cur = acc
        while ran + policy.m <= window:
            ran += policy.m
            t_end = ran
            new_acc = a_inf - (a_inf - a_post) * math.exp(-t_end / model.tau)
            # trapezoid over this eval interval
            area += 0.5 * (cur + new_acc) * policy.m
            cur = new_acc
            obs = cur + (rng.normal(0.0, _canary_se(cur, n_canary)) if noise else 0.0)
            if stopper.update(obs):
                break
        acc = cur
        area += acc * (window - ran)
        clock = next_batch
        burst_total += ran
        bursts += 1
        reasons[stopper.reason or "window_end"] = reasons.get(stopper.reason or "window_end", 0) + 1

    if clock < horizon:
        area += acc * (horizon - clock)
    return LifetimeResult(
        lifetime_mean_acc=area / max(horizon, 1),
        final_acc=acc,
        worst_dip=clean_acc - worst,
        burst_batches=float(burst_total),
        spares_used=remap_capacity - spares_left,
        bursts_run=float(bursts),
        stop_reasons=reasons,
    )


def simulate_lifetime(
    policy_params: PolicyParams,
    recovery_models: dict[float, RecoveryModel],
    wave_schedule: Sequence[Wave],
    remap_capacity_model: float,
    horizon: int | None = None,
    clean_acc: float = 90.0,
    remap_efficiency: float = 1.0,
    n_canary: int = 1000,
    noise: bool = True,
    n_rep: int = 20,
    seed: int = 0,
) -> LifetimeResult:
    """Simulate one deployment lifetime (see the module docstring for every
    modeling assumption) and return the mean over `n_rep` noise realizations
    (a single noiseless run if `noise=False`).

    `remap_capacity_model` is total spare capacity in fraction-of-cells units.
    `horizon` defaults to the last wave + 1000 batches.
    """
    model = recovery_models[policy_params.eta]
    waves = list(wave_schedule)
    if horizon is None:
        horizon = (max(w.batch for w in waves) if waves else 0) + 1000
    reps = max(n_rep, 1) if noise else 1
    rng = np.random.default_rng(seed)
    runs = [
        _simulate_once(
            policy_params,
            model,
            waves,
            horizon,
            clean_acc,
            remap_capacity_model,
            remap_efficiency,
            n_canary,
            noise,
            rng,
        )
        for _ in range(reps)
    ]
    reasons: dict[str, float] = {}
    for r in runs:
        for key, v in r.stop_reasons.items():
            reasons[key] = reasons.get(key, 0) + v / reps
    return LifetimeResult(
        lifetime_mean_acc=float(np.mean([r.lifetime_mean_acc for r in runs])),
        final_acc=float(np.mean([r.final_acc for r in runs])),
        worst_dip=float(np.mean([r.worst_dip for r in runs])),
        burst_batches=float(np.mean([r.burst_batches for r in runs])),
        spares_used=float(np.mean([r.spares_used for r in runs])),
        bursts_run=float(np.mean([r.bursts_run for r in runs])),
        stop_reasons=reasons,
    )


# --------------------------------------------------------------------------
# Grid search
# --------------------------------------------------------------------------


@dataclass
class GridPoint:
    params: PolicyParams
    result: LifetimeResult
    objective: float = 0.0


@dataclass
class GridSearchResult:
    points: list[GridPoint]
    pareto: list[GridPoint]
    chosen: GridPoint
    lam: float

    def chosen_dict(self) -> dict:
        return {
            "params": asdict(self.chosen.params),
            "lifetime_mean_acc": self.chosen.result.lifetime_mean_acc,
            "burst_batches": self.chosen.result.burst_batches,
            "objective": self.chosen.objective,
        }


def pareto_front(points: Sequence[GridPoint]) -> list[GridPoint]:
    """Non-dominated subset: maximize lifetime-mean acc, minimize burst batches."""
    front = []
    for p in points:
        a, b = p.result.lifetime_mean_acc, p.result.burst_batches
        dominated = any(
            q.result.lifetime_mean_acc >= a
            and q.result.burst_batches <= b
            and (q.result.lifetime_mean_acc > a or q.result.burst_batches < b)
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.result.burst_batches)


def grid_search(
    recovery_models: dict[float, RecoveryModel],
    wave_schedule: Sequence[Wave],
    remap_capacity_model: float,
    lam: float,
    etas: Sequence[float] | None = None,
    deltas: Sequence[float] = (0.5, 2.0),
    kappas: Sequence[float] = (0.05, 0.1, 0.25),
    ks: Sequence[int] = (3, 4, 6),
    ms: Sequence[int] = (25,),
    t_maxs: Sequence[int] = (500, 1000, 2000),
    pacings: Sequence[str] = ("greedy", "paced"),
    **sim_kwargs,
) -> GridSearchResult:
    """Exhaustive grid over (eta, delta, kappa, k, m, t_max, pacing). Objective:
    lifetime-mean accuracy - lam * total burst batches (lam in accuracy points
    per batch, chosen up front). Returns all points, the Pareto set over
    (lifetime-mean acc, burst batches) and the objective-maximizing point (which
    is always on the Pareto set)."""
    etas = list(etas) if etas is not None else sorted(recovery_models)
    points = []
    for eta, delta, kappa, k, m, t_max, pacing in itertools.product(
        etas, deltas, kappas, ks, ms, t_maxs, pacings
    ):
        params = PolicyParams(eta, delta, kappa, int(k), int(m), int(t_max), pacing)
        res = simulate_lifetime(
            params, recovery_models, wave_schedule, remap_capacity_model, **sim_kwargs
        )
        points.append(GridPoint(params, res, res.lifetime_mean_acc - lam * res.burst_batches))
    chosen = max(points, key=lambda p: p.objective)
    return GridSearchResult(points=points, pareto=pareto_front(points), chosen=chosen, lam=lam)
