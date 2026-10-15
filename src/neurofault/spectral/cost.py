"""Energy / latency proxies for counted read cost (cost.py).

Why a proxy and not a simulation: the experiments compare methods by COUNTED reads (ReadCost) and only
convert to energy/latency to ask whether a win survives at plausible hardware numbers. Every
parameter below is an ASSUMPTION taken from literature ranges (no device data),
recorded as ParamRecord(status="assumed") so it is printed into every result's params.json and
can be swept. Never quote an energy number without the sweep.

Model (deliberately simple):
- energy = sum(count * per-event energy). An MVM costs the array's static read energy; each ADC
  conversion, DAC drive, single-cell read, write and calibration image has its own constant.
- latency = serial analog work. MVM latency includes its own readout settle (t_mvm_s), so ADC
  conversions add ENERGY only; single-cell reads are column-parallel (read_parallelism per
  cycle); writes and calibration images are serial.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from neurofault.spectral.types import ParamRecord, ReadCost

_SRC = "literature range midpoint"


@dataclass(frozen=True)
class CostModel:
    """All fields are [assumed]; see `param_records()` for the machine-readable ledger."""

    e_mvm_pj: float = 20.0  # array static read energy per MVM (one tile, one vector)
    e_adc_pj: float = 5.0  # per 8-bit conversion (range 3-10 pJ)
    e_dac_pj: float = 1.0  # per input drive
    e_cell_read_pj: float = 0.5  # per individually read cell (full readout path)
    e_write_pj: float = 100.0  # per programmed cell (endurance matters more than energy)
    e_image_pj: float = 1.0e6  # one calibration image through the network (AdaBS) = 1 uJ
    t_mvm_s: float = 50e-9  # one MVM incl. TIA settle to 8 b (40-60 ns at GBW 100 MHz)
    t_cell_read_s: float = 50e-9  # one parallel cell-read cycle
    read_parallelism: int = 256  # cells read per cycle (column-parallel)
    t_write_s: float = 1e-6  # per write pulse (serial)
    t_image_s: float = 1e-3  # per calibration image

    def energy(self, cost: ReadCost) -> float:
        """Energy proxy in joules."""
        pj = (
            (cost.mvm_fwd + cost.mvm_bwd) * self.e_mvm_pj
            + cost.adc * self.e_adc_pj
            + cost.dac * self.e_dac_pj
            + cost.cell_reads * self.e_cell_read_pj
            + cost.writes * self.e_write_pj
            + cost.images * self.e_image_pj
        )
        return pj * 1e-12

    def latency(self, cost: ReadCost) -> float:
        """Latency proxy in seconds (serial analog work; see module docstring)."""
        return (
            (cost.mvm_fwd + cost.mvm_bwd) * self.t_mvm_s
            + cost.cell_reads / self.read_parallelism * self.t_cell_read_s
            + cost.writes * self.t_write_s
            + cost.images * self.t_image_s
        )

    def edp(self, cost: ReadCost) -> float:
        """Energy-delay product (J*s), the scalar the C' bake-off ranks by (before area)."""
        return self.energy(cost) * self.latency(cost)

    def param_records(self) -> dict[str, ParamRecord]:
        """Every parameter marked assumed, for params.json."""
        return {
            f.name: ParamRecord(
                value=getattr(self, f.name),
                status="assumed",
                source=_SRC,
                verified=False,
                sweep=None,
            )
            for f in fields(self)
        }
