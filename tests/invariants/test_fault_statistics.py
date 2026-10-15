"""Statistical validity of the fault models - do they actually produce the
distribution their own math says they should, not just "some cells become
stuck"? Marked `statistical`: goodness-of-fit checks at a fixed seed and a
disclosed significance level, not flaky retries.
"""

from __future__ import annotations

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from neurofault.config import FaultConfig
from neurofault.crossbar.health_monitor import HealthMonitor
from neurofault.faults.injection import _select_fault_indices, inject_faults
from neurofault.faults.reliability import (
    arrhenius_acceleration_factor,
    weibull_conditional_failure_probability,
)
from tests.unit.fakes import make_handle

pytestmark = pytest.mark.statistical


def test_density_model_matches_target_density_within_binomial_confidence():
    """density=d on an RxC array must actually stick close to d*rows*cols
    cells, not just "a plausible-looking number" - scipy.stats.binomtest
    against the exact binomial model density picking is supposed to follow."""
    torch.manual_seed(0)
    rows, cols = 64, 64
    density = 0.15
    handle = make_handle(shape=(rows, cols))
    monitor = HealthMonitor(handle)
    fault_config = FaultConfig(density=density)

    injected = inject_faults(handle, monitor, fault_config)

    result = stats.binomtest(injected, rows * cols, density)
    assert result.pvalue > 0.001  # fixed seed; a real deviation would fail far harder than this


def test_weibull_empirical_survival_matches_theoretical_cdf():
    """Simulate a large population of independent cells through many
    statistical stuck-at cycles; compare the empirical survival curve to the
    theoretical Weibull CDF via a one-sample KS test."""
    torch.manual_seed(1)
    shape, scale = 2.0, 200.0
    n_cells = 2000
    # Long enough that censoring (cells still alive at window end) doesn't
    # bias the empirical distribution vs. the unconditional CDF the KS test
    # compares against: S(600) = exp(-(600/200)^2) = exp(-9) ~ negligible.
    n_cycles = 600

    cycles_survived = torch.zeros(n_cells)
    alive = torch.ones(n_cells, dtype=torch.bool)
    for cycle in range(1, n_cycles + 1):
        cycles_before = torch.full((n_cells,), float(cycle - 1))
        delta = torch.ones(n_cells)
        fail_prob = weibull_conditional_failure_probability(cycles_before, delta, shape, scale)
        newly_failed = torch.bernoulli(fail_prob).bool() & alive
        cycles_survived[newly_failed] = cycle
        alive &= ~newly_failed

    failure_times = cycles_survived[~alive].numpy()
    assert int((~alive).sum().item()) > int(n_cells * 0.99)  # censoring must be negligible

    def weibull_cdf(x):
        return stats.weibull_min(shape, scale=scale).cdf(x)

    _ks_stat, p_value = stats.kstest(failure_times, weibull_cdf)
    assert p_value > 0.01, (
        f"empirical failure times diverge from theoretical Weibull CDF (KS p={p_value})"
    )


def test_exponential_empirical_survival_matches_theoretical_cdf():
    """Window must run long enough that censoring (cells still alive when
    the window ends) doesn't bias the empirical failure-time distribution
    the KS test compares against the unconditional exponential CDF - at
    rate=0.01 (mean life 100 cycles), 1500 cycles leaves P(survive)=exp(-15)
    ~ negligible, not a real modeling assumption change."""
    torch.manual_seed(2)
    rate = 0.01
    n_cells = 2000
    n_cycles = 1500

    cycles_survived = torch.zeros(n_cells)
    alive = torch.ones(n_cells, dtype=torch.bool)
    for cycle in range(1, n_cycles + 1):
        delta = torch.ones(n_cells)
        fail_prob = 1.0 - torch.exp(-rate * delta)
        newly_failed = torch.bernoulli(fail_prob).bool() & alive
        cycles_survived[newly_failed] = cycle
        alive &= ~newly_failed

    failure_times = cycles_survived[~alive].numpy()
    assert int((~alive).sum().item()) > int(n_cells * 0.99)  # censoring must be negligible

    _ks_stat, p_value = stats.kstest(failure_times, lambda x: stats.expon(scale=1 / rate).cdf(x))
    assert p_value > 0.01, (
        f"empirical failure times diverge from theoretical exponential CDF (p={p_value})"
    )


@given(
    t=st.floats(min_value=200.0, max_value=400.0, allow_nan=False),
    t_ref=st.floats(min_value=200.0, max_value=400.0, allow_nan=False),
    ea=st.floats(min_value=0.05, max_value=2.0, allow_nan=False),
)
@settings(max_examples=200, deadline=None)
def test_arrhenius_ordering_invariants_hold_for_any_physical_inputs(t, t_ref, ea):
    """Property-based sweep, not 3 hardcoded cases: for ANY physically valid
    (T, T_ref, Ea), the acceleration factor must respect these orderings."""
    factor = arrhenius_acceleration_factor(t, t_ref, ea)

    if t > t_ref:
        assert factor > 1.0
    elif t < t_ref:
        assert factor < 1.0
    else:
        assert math.isclose(factor, 1.0, rel_tol=1e-9)


@given(
    ea_base=st.floats(min_value=0.05, max_value=1.5),
    ea_delta=st.floats(min_value=0.01, max_value=0.5),
)
@settings(max_examples=50, deadline=None)
def test_arrhenius_factor_increases_with_activation_energy_above_reference(ea_base, ea_delta):
    """Higher activation energy -> stronger temperature sensitivity -> a
    strictly larger acceleration factor above the reference temperature.
    ea_delta is always positive, so ea_base+ea_delta > ea_base is guaranteed
    by construction - no boundary-overlap flake possible."""
    factor_low = arrhenius_acceleration_factor(350.0, 300.0, ea_base)
    factor_high = arrhenius_acceleration_factor(350.0, 300.0, ea_base + ea_delta)
    assert factor_high > factor_low


def test_clustered_distribution_has_lower_mean_pairwise_distance_than_random():
    """A real spatial-shape check, not just "the right count got picked" -
    clustered faults must be measurably more spatially concentrated than
    random ones at the same density."""
    torch.manual_seed(3)
    rows, cols, num_faults = 50, 50, 200

    def mean_pairwise_distance(distribution: str) -> float:
        indices = _select_fault_indices(rows, cols, num_faults, distribution, torch.device("cpu"))
        r = (indices // cols).float()
        c = (indices % cols).float()
        pts = torch.stack([r, c], dim=1)
        diffs = pts.unsqueeze(0) - pts.unsqueeze(1)
        dists = torch.sqrt((diffs**2).sum(dim=-1) + 1e-12)
        return dists.mean().item()

    clustered_distances = [mean_pairwise_distance("clustered") for _ in range(10)]
    random_distances = [mean_pairwise_distance("random") for _ in range(10)]

    assert sum(clustered_distances) / len(clustered_distances) < sum(random_distances) / len(
        random_distances
    )


def test_gradient_distribution_correlates_fault_density_with_position():
    """gradient must produce a real, statistically significant positive
    correlation between position and selection weight, not just a vaguely
    biased-looking sample."""
    torch.manual_seed(4)
    rows, cols, num_faults = 60, 60, 1000

    indices = _select_fault_indices(rows, cols, num_faults, "gradient", torch.device("cpu"))
    r = (indices // cols).numpy()
    c = (indices % cols).numpy()
    position_score = r / rows + c / cols

    # gradient favors LOW position_score (weights = 1 - distance/2) - assert
    # the sample's position_score distribution skews low relative to a
    # uniform-random sample, via a one-sided Mann-Whitney U test (a real
    # statistical shape check, not just eyeballing the mean).
    random_indices = _select_fault_indices(rows, cols, num_faults, "random", torch.device("cpu"))
    random_r = (random_indices // cols).numpy()
    random_c = (random_indices % cols).numpy()
    random_score = random_r / rows + random_c / cols

    _u_stat, p_value = stats.mannwhitneyu(position_score, random_score, alternative="less")
    assert p_value < 0.01, "gradient-selected positions are not skewed toward low position_score"
