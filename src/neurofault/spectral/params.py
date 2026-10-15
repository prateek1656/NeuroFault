"""Every numeric constant behind the drift / fault / noise models.

WHY a registry: results are only interpretable next to the parameters that
produced them, and "sourced" vs "assumed" must be visible per number so no
assumed knob is ever summarised without its sweep. Models read their defaults
from here and run directories dump `dump_params()` to params.json.

`verified=True` means the constant was read in the installed AIHWKit 1.1.0
source (or repo source) during the build of this module, not copied from the
plan or a paper abstract. Paper-only numbers are `verified=False`.
"""

from __future__ import annotations

from neurofault.spectral.types import ParamRecord

_PCM = "aihwkit 1.1.0 inference/noise/pcm.py PCMLikeNoiseModel"
_CMO = "aihwkit 1.1.0 inference/noise/reram.py ReRamCMONoiseModel"


def _src(value, source: str, verified: bool = True) -> ParamRecord:
    return ParamRecord(value=value, status="sourced", source=source, verified=verified)


def _asm(value, sweep: tuple, source: str) -> ParamRecord:
    return ParamRecord(value=value, status="assumed", source=source, verified=False, sweep=sweep)


PARAMS: dict[str, ParamRecord] = {
    # ---- timing -----------------------------------------------------------------
    "t0_s": _src(20.0, f"{_PCM}.__init__ t_0=20.0 (generate/apply_drift use t = t_inf + t_0)"),
    "t0_joshi_s": _src(25.0, "Joshi et al. Nat. Commun. 11, 2473 (2020) ", False),
    "t_read_pcm_s": _src(250e-9, f"{_PCM}.__init__ t_read=250e-9"),
    # ---- D1 uniform power law ---------------------------------------------------------------
    "d1_nu": _src(0.06, "Joshi 2020: 'nu ~ 0.06 on average' (paper text not re-read)", False),
    # ---- D2 IBM PCM per-device nu (generate_drift_coefficients, lines ~121-130) ---------------
    "d2_mu_slope": _src(-0.0155, f"{_PCM}.generate_drift_coefficients mu_drift"),
    "d2_mu_intercept": _src(0.0244, f"{_PCM}.generate_drift_coefficients mu_drift"),
    "d2_mu_clip": _src((0.049, 0.1), f"{_PCM}.generate_drift_coefficients mu_drift.clamp"),
    "d2_sigma_slope": _src(-0.0125, f"{_PCM}.generate_drift_coefficients sig_drift"),
    "d2_sigma_intercept": _src(-0.0059, f"{_PCM}.generate_drift_coefficients sig_drift"),
    "d2_sigma_clip": _src((0.008, 0.045), f"{_PCM}.generate_drift_coefficients sig_drift.clamp"),
    "d2_g_rel_floor": _src(1e-7, f"{_PCM} _ZERO_CLIP (g_relative clamp min), natural log"),
    "d2_g_ref_uS": _src(25.0, f"{_PCM} ref g_max via SinglePairConductanceConverter default"),
    # ---- PCM read / programming noise (consumed by noise.py) -----------------------------------
    "pcm_prog_coeff": _src(
        [0.26348, 1.9650, -1.1731], f"{_PCM}.__init__ prog_coeff (g_max ref 25)"
    ),
    "pcm_1f_q": _src(0.0088, f"{_PCM}.apply_drift_noise_to_conductance q_s numerator"),
    "pcm_1f_exp": _src(0.65, f"{_PCM}.apply_drift_noise_to_conductance (g/gmax)^0.65"),
    "pcm_1f_q_max": _src(0.2, f"{_PCM}.apply_drift_noise_to_conductance q_s clamp max"),
    # ---- D3 structured nu -----------------------------------------------------------------------
    "d3_rho": _asm(1.0, (0.0, 0.25, 0.5, 1.0, 2.0, 4.0), "D3; no measured correlation"),
    "d3_rank_r": _asm(2, (1, 2, 4), "D3 rank-r cosine field"),
    "d3_hotspot_width_frac": _asm(
        0.2, (0.1, 0.2, 0.4), "gaussian sigma / max(m, n); builder choice"
    ),
    # ---- D4 temperature -------------------------------------------------------------------
    "d4_Ea_eV": _asm(0.5, (0.2, 0.5, 1.0), "D4(i); no verified source"),
    "d4_T_ref_K": _asm(300.0, (300.0,), "room temperature; FaultConfig default"),
    "d4_T_K": _asm(330.0, (310.0, 330.0, 360.0), "operating temperature; builder choice"),
    "d4_beta_per_K": _asm(1e-3, (5e-4, 1e-3, 3e-3), "0.05/0.1/0.3 %/K gain coefficient"),
    "d4_dT_amp_K": _asm(20.0, (5.0, 20.0, 40.0), "D4(ii)"),
    # ---- D5 RRAM relaxation ------------------------------------------------------------------
    "d5a_decay_mean_uS": _src([-0.08900206], f"{_CMO}.__init__ decay_dict['mean']"),
    "d5a_decay_std": _src(
        [0.04201137, 0.41183342], f"{_CMO}.__init__ decay_dict['std'] = [slope_ln_t, intercept] uS"
    ),
    "d5a_read_K": _src(0.0277316483, f"{_CMO}.__init__ read_dict['K']"),
    "d5a_t_read_s": _src(1e-6, f"{_CMO}.__init__ read_dict['t_read']"),
    "d5b_c": _asm(0.01, (0.005, 0.01, 0.02), "mean-reversion rate lambda = c ln(1 + t/t0)"),
    # ---- D7 periphery gain --------------------------------------------------------------------
    "d7_sigma_p": _asm(0.01, (0.005, 0.01, 0.02), "per-line gain spread, D7"),
    # ---- D0 legacy rail shift (repo behaviour) -----------------------------------------------
    "d0_drift_magnitude": _src(0.05, "neurofault/config.py FaultConfig.drift_magnitude default"),
    # ---- faults -------------------------------------------------------------------------------
    "fault_k": _asm(8, (1, 2, 4, 8, 16, 32, 64), "fault count grid"),
    "fault_cluster_blob": _src("3x3", "faults/injection.py::_select_fault_indices clustered"),
}


def dump_params() -> dict:
    """JSON-serialisable provenance dump (written to params.json by every run)."""
    out = {}
    for k, r in PARAMS.items():
        out[k] = {
            "value": r.value,
            "status": r.status,
            "source": r.source,
            "verified": r.verified,
            "sweep": None if r.sweep is None else list(r.sweep),
        }
    return out


def val(key: str):
    """Value of a registered parameter (single entry point so models stay in sync)."""
    return PARAMS[key].value
