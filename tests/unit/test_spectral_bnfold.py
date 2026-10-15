import numpy as np
import pandas as pd
import torch

from neurofault.spectral.bnfold import add_gain_ci, tile_origins


def test_tile_origins_skips_thin_edges():
    mats = {"a": torch.zeros(300, 600), "b": torch.zeros(10, 20)}
    t = tile_origins(mats, ["a", "b"], 256, min_dim=16)
    # a: rows 256+44 (44>=16 ok), cols 256,256,88 -> 2x3 tiles; b: both sides < 16 on rows -> none
    assert len(t) == 6
    assert ("a", 256, 512, 44, 88) in t
    assert all(x[0] == "a" for x in t)


def test_add_gain_ci_groups_and_sign():
    rows = []
    for ds in ("c10", "c100"):
        for t in (1.0, 2.0):
            for s in (1, 2, 3):
                for sc in range(4):
                    rows.append(
                        {
                            "dataset": ds,
                            "t_s": t,
                            "ckpt_seed": s,
                            "gain_m4_minus_b8_pt": (5.0 if ds == "c10" else 0.0) + 0.1 * sc,
                        }
                    )
    out = add_gain_ci(pd.DataFrame(rows), n_boot=200)
    c10 = out[out.dataset == "c10"].gain_ci_lower_pt
    assert (c10 > 4.5).all() and c10.nunique() <= 2
    assert (out[out.dataset == "c100"].gain_ci_lower_pt < 0.5).all()
    assert not np.isnan(out.gain_ci_lower_pt).any()
