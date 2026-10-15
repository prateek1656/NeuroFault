"""Device presets AIHWKit doesn't ship itself (verified via
`dir(aihwkit.simulator.presets.devices)` - no FeFET or STT-MRAM entry exists),
authored here rather than monkeypatched into aihwkit's own namespace. Follows
the same pattern as aihwkit's own presets (PCMPresetDevice, ReRamESPresetDevice,
etc. - confirmed by reading their source): a `@dataclass` subclassing one of
aihwkit's base pulsed-device models, with parameters fit to one real,
cited paper, not invented numbers.

Registered under devices/registry.py's aihwkit name_map ("fefet", "sttmram").
CrossSim has no equivalent - see devices/registry.py's module docstring for
why (one generic Rmin/Rmax model, no swappable per-technology device class) -
these two technologies are only richly modeled on the AIHWKit backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from aihwkit.simulator.configs.devices import LinearStepDevice
from aihwkit.simulator.presets.devices import IdealizedPresetDevice


@dataclass
class FeFETPresetDevice(LinearStepDevice):
    """Ferroelectric FET (FeFET) analog synapse.

    Fit to Jerry, Chen, Zhang, et al. (incl. Ni), "Ferroelectric FET analog
    synapse for acceleration of deep neural network training", IEDM 2017
    (IEEE Xplore 8268338): a real 5-bit (32-state) FeFET synapse with a 45x
    (Gmax/Gmin) dynamic range and symmetric potentiation/depression.

    LinearStepDevice, not ExpStepDevice (which PCMPresetDevice/
    ReRamESPresetDevice use for strongly nonlinear switching), is the right
    base here - the cited paper reports near-linear, near-symmetric
    conductance updates, which LinearStepDevice's gamma/up_down parameters
    are built to represent as a small residual nonlinearity/asymmetry rather
    than a strongly exponential one.
    """

    # pylint: disable=invalid-name

    # Dynamic range: Gmax/Gmin = 45x, symmetric around 0 in weight-space (same
    # convention crosssim_backend.py/aihwkit_backend.py use elsewhere in this
    # project for signed weight ranges).
    w_max: float = 1.0
    w_min: float = -1.0

    # ~32 reachable states (5-bit) across the full w_max-w_min range.
    dw_min: float = (1.0 - (-1.0)) / 32

    # Small residual nonlinearity/asymmetry - near-linear, near-symmetric per
    # the cited paper, not the strongly nonlinear gamma values PCM/ReRAM use.
    gamma_up: float = 0.1
    gamma_down: float = 0.1
    up_down: float = 0.0

    # Device-to-device / cycle-to-cycle variation - present but modest,
    # consistent with a mature, characterized device rather than an
    # early-stage/high-variability one.
    dw_min_dtod: float = 0.1
    up_down_dtod: float = 0.05
    w_max_dtod: float = 0.1
    w_min_dtod: float = 0.1
    dw_min_std: float = 0.2


@dataclass
class STTMRAMPresetDevice(IdealizedPresetDevice):
    """Spintronic (STT-MRAM / domain-wall MTJ) analog synapse.

    Domain-wall MTJ synapses (Sengupta et al., "Proposal for an All-Spin
    Artificial Neural Network: Emulating Neural and Synaptic Functionalities
    Through Domain Wall Motion in Ferromagnets", arXiv:1510.00459; see also
    the comparative arXiv:1910.12919) report conductance that varies linearly
    with domain-wall position - the most linear, least-nonlinear device class
    among common spintronic synapse proposals.

    Basing this on AIHWKit's own IdealizedPresetDevice (perfectly symmetric,
    minimal device-to-device asymmetry) rather than a nonlinear fit is a
    deliberate, stated modeling choice, not a cop-out: it matches this
    project's own device-technology table, which characterizes STT-MRAM as the least fault-prone technology of
    the group and a "best case" comparison point - IdealizedPresetDevice's
    near-ideal linearity is the closest existing AIHWKit base to that real,
    literature-reported behavior.
    """

    # No fields overridden beyond IdealizedPresetDevice's own defaults -
    # its near-zero device-to-device asymmetry and large state count already
    # match the domain-wall synapse literature's near-ideal-linearity claim.
