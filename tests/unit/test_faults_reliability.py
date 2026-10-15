import math

import torch

from neurofault.faults.reliability import (
    arrhenius_acceleration_factor,
    exponential_conditional_failure_probability,
    weibull_conditional_failure_probability,
)


def test_weibull_cdf_at_scale_is_shape_independent():
    """Exact invariant: CDF at n=scale is 1-e^-1 for ANY shape - a scale-only
    fact, not an approximation, useful as a precise regression test."""
    scale = 1000.0
    for shape in (0.5, 1.0, 2.0, 5.0):
        prob = weibull_conditional_failure_probability(
            cycles_before=torch.tensor(0.0),
            delta_cycles=torch.tensor(scale),
            shape=shape,
            scale=scale,
        )
        assert math.isclose(prob.item(), 1 - math.exp(-1), abs_tol=1e-6)


def test_weibull_probability_is_zero_when_no_time_has_passed():
    prob = weibull_conditional_failure_probability(
        cycles_before=torch.tensor(500.0), delta_cycles=torch.tensor(0.0), shape=2.0, scale=1000.0
    )
    assert math.isclose(prob.item(), 0.0, abs_tol=1e-9)


def test_weibull_probability_increases_monotonically_with_delta_cycles():
    cycles_before = torch.tensor(0.0)
    deltas = [10.0, 100.0, 500.0, 1000.0]
    probs = [
        weibull_conditional_failure_probability(cycles_before, torch.tensor(d), 2.0, 1000.0).item()
        for d in deltas
    ]
    assert probs == sorted(probs)


def test_exponential_probability_zero_when_no_time_has_passed():
    prob = exponential_conditional_failure_probability(delta_cycles=torch.tensor(0.0), rate=0.01)
    assert math.isclose(prob.item(), 0.0, abs_tol=1e-9)


def test_exponential_probability_is_memoryless():
    """Conditional failure probability doesn't depend on prior cycles at
    all - only the elapsed interval matters."""
    rate = 0.01
    delta = torch.tensor(50.0)
    prob_a = exponential_conditional_failure_probability(delta, rate)
    prob_b = exponential_conditional_failure_probability(delta, rate)  # no "cycles_before" input
    assert math.isclose(prob_a.item(), prob_b.item())
    # float32 (torch default) vs float64 (Python math) precision - not exact.
    assert math.isclose(prob_a.item(), 1 - math.exp(-rate * 50.0), abs_tol=1e-6)


def test_arrhenius_factor_is_exactly_one_at_reference_temperature():
    for activation_energy in (0.3, 0.7, 1.1):
        factor = arrhenius_acceleration_factor(
            temperature_kelvin=300.0,
            reference_temperature_kelvin=300.0,
            activation_energy_ev=activation_energy,
        )
        assert math.isclose(factor, 1.0, abs_tol=1e-9)


def test_arrhenius_factor_exceeds_one_above_reference_temperature():
    factor = arrhenius_acceleration_factor(
        temperature_kelvin=350.0, reference_temperature_kelvin=300.0, activation_energy_ev=0.7
    )
    assert factor > 1.0


def test_arrhenius_factor_below_one_under_reference_temperature():
    factor = arrhenius_acceleration_factor(
        temperature_kelvin=250.0, reference_temperature_kelvin=300.0, activation_energy_ev=0.7
    )
    assert factor < 1.0
