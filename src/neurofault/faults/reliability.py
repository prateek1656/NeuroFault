"""Statistical time-to-failure models for fault injection - real applied
reliability engineering (Weibull wear-out, exponential constant-hazard,
Arrhenius temperature acceleration), not just another spatial distribution
option. faults/injection.py's `distribution` param (random/clustered/
gradient) answers "which cells, given how many must fail this round" - a
spatial question. The functions here answer "how many fail this round, and
does this cell's own history make it more or less likely" - a temporal/
physical question. Deliberately orthogonal to spatial distribution, not
combined with it in this pass.

`cycles`/`delta_cycles` here are a simulation-time proxy - one unit per
inject_faults() call a cell survives - not a count of literal physical
read/write operations on real hardware.
"""

from __future__ import annotations

import math

import torch


def weibull_conditional_failure_probability(
    cycles_before: torch.Tensor,
    delta_cycles: torch.Tensor,
    shape: float,
    scale: float,
) -> torch.Tensor:
    """P(fails during this interval | survived to cycles_before), derived
    from the Weibull survival function S(n) = exp(-(n/scale)^shape):

        P = 1 - S(cycles_before + delta_cycles) / S(cycles_before)
          = 1 - exp((cycles_before/scale)^shape - ((cycles_before+delta_cycles)/scale)^shape)
    """
    prior_exponent = (cycles_before / scale) ** shape
    after_exponent = ((cycles_before + delta_cycles) / scale) ** shape
    return 1.0 - torch.exp(prior_exponent - after_exponent)


def exponential_conditional_failure_probability(
    delta_cycles: torch.Tensor, rate: float
) -> torch.Tensor:
    """Memoryless - conditional failure probability doesn't depend on prior
    history, only the elapsed interval: P = 1 - exp(-rate * delta_cycles)."""
    return 1.0 - torch.exp(-rate * delta_cycles)


def arrhenius_acceleration_factor(
    temperature_kelvin: float,
    reference_temperature_kelvin: float,
    activation_energy_ev: float,
) -> float:
    """Standard Arrhenius acceleration factor, as used in JEDEC-style
    accelerated life testing:

        AF = exp(Ea/k_B * (1/T_ref - 1/T))

    AF > 1 when T > T_ref (failures accelerate faster than at the reference
    temperature the base weibull_scale/exponential_rate were characterized
    at); AF == 1 exactly when T == T_ref, for any activation energy.
    """
    boltzmann_ev_per_kelvin = 8.617333262e-5
    return math.exp(
        activation_energy_ev
        / boltzmann_ev_per_kelvin
        * (1.0 / reference_temperature_kelvin - 1.0 / temperature_kelvin)
    )
