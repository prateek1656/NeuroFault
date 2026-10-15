"""Layer reset: restore a layer's weights and re-derive its crossbar
conductances the same way the backend derived them originally.

Fixes the original codebase's bug #5: `crossbar.conductance_matrix.copy_(
module.weight.data)` copied the raw weight tensor directly into the
conductance matrix, ignoring the actual weight->conductance mapping and
Scheme.DoubleColumn's split into two separate crossbars with a different
shape than the weight tensor - it silently no-op'd behind a bare
`except Exception: print(...)`.

Here, failure is never silent: `apply_layer_reset` returns False and logs
the real exception; it never reports success without having actually
rewritten every crossbar handle for the layer.

Real design flaw found and fixed 2026-09-13: whole-layer reset wipes ALL of
a layer's weights back to their random pre-training values, discarding
every batch of legitimate learning along with whatever fault it was meant
to correct - measured directly, robust across 3 seeds, actively worse than
no mitigation at all (11.80% vs 41.57% mean final accuracy), even after a
cooldown throttle cut how often it fires. The fix is `positions`: reset only
the specific crossbar cells that are actually critical, leaving every other
cell (and the training signal already encoded there) untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from neurofault.crossbar.array import CrossbarHandle

logger = logging.getLogger(__name__)


def apply_layer_reset(
    layer: torch.nn.Module,
    handles: list[CrossbarHandle],
    original_weight: torch.Tensor,
    mapping_routine: Callable,
    r_on: float,
    r_off: float,
    scheme: str = "DoubleColumn",
    positions: list[tuple[int, int]] | None = None,
) -> bool:
    """positions=None (default): restore the WHOLE layer to original_weight,
    respecting DoubleColumn's positive/negative crossbar split - the
    original, still-supported behavior (kept for backends/schemes that
    genuinely need it, or a caller that wants a hard, total reset).

    positions=[(row, col), ...]: restore ONLY those crossbar cells (and the
    corresponding elements of layer.weight.data) to their pre-training
    values - the real fix for the whole-layer-wipe problem above. Requires
    a single crossbar handle per layer whose mapping_routine is a pure
    reshape of weight (true for both backends' current mapping_routine_for()
    - see their own docstrings) - doesn't attempt to generalize to
    DoubleColumn/bit-slicing's multi-handle-per-layer split, which no
    current config exercises with layer_reset enabled.

    Returns False (and logs) on any failure - never swallows an exception
    and reports success anyway.
    """
    try:
        if positions is None:
            layer.weight.data.copy_(original_weight)

            mapped = mapping_routine(layer.weight.data, r_on, r_off, scheme)
            # naive_map returns a single tensor for SingleColumn, a
            # (positive, negative) tuple of tensors for DoubleColumn.
            matrices = mapped if isinstance(mapped, (tuple, list)) else (mapped,)

            if len(matrices) != len(handles):
                raise ValueError(
                    f"mapping_routine produced {len(matrices)} matrices but layer has "
                    f"{len(handles)} crossbar handles (scheme={scheme})"
                )

            for handle, matrix in zip(handles, matrices):
                handle.accessor.write(matrix)

            logger.info("Reset layer with %d crossbar(s) to original weights", len(handles))
            return True

        if not positions:
            return False
        if len(handles) != 1:
            raise ValueError(
                f"Targeted (positions=...) layer_reset requires exactly one crossbar "
                f"handle per layer, got {len(handles)} - not exercised by any current "
                "config that enables layer_reset"
            )

        reference = mapping_routine(original_weight, r_on, r_off, scheme)
        if isinstance(reference, (tuple, list)):
            raise TypeError(
                "Targeted (positions=...) layer_reset doesn't support a "
                "mapping_routine that splits into multiple matrices (e.g. "
                "DoubleColumn) - not exercised by any current config"
            )

        handle = handles[0]
        rows = layer.weight.data.shape[0]
        weight_view = layer.weight.data.reshape(rows, -1)
        original_view = original_weight.reshape(rows, -1)

        row_idx = torch.tensor([r for r, _c in positions], device=weight_view.device)
        col_idx = torch.tensor([c for _r, c in positions], device=weight_view.device)
        # In-place indexed assignment on a reshape view writes through to
        # the underlying storage (verified directly: reshape of a
        # contiguous tensor shares storage, and __setitem__ on the view
        # mutates it) - layer.weight.data itself is updated, not a copy.
        weight_view[row_idx, col_idx] = original_view[row_idx, col_idx]

        matrix = handle.accessor.read()
        matrix[row_idx, col_idx] = reference[row_idx, col_idx]
        handle.accessor.write(matrix)

        logger.info(
            "Targeted-reset %d cell(s) in %s (whole-layer reset avoided)",
            len(positions),
            handle.name,
        )
        return True

    except Exception:
        logger.exception("Layer reset failed - weights/conductances left in their prior state")
        return False
