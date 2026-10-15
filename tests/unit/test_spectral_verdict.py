"""Verdict engine: schema validation of every criteria file + toy-frame evaluation, and the
harness provenance writer (kept together: both are the 'traceable number' machinery)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from neurofault.spectral import harness
from neurofault.spectral import verdict as V

CRITERIA_DIR = Path(__file__).resolve().parents[2] / "experiments" / "spectral" / "criteria"
FILES = sorted(CRITERIA_DIR.glob("*.yaml"))


def _crit(criteria: list[dict], overall_pass: str = "all") -> V.Criteria:
    raw = {"schema_version": 1, "id": "toy", "title": "t", "source_plan": "p", "frozen": "x",
           "overall_pass": overall_pass, "criteria": criteria}  # fmt: skip
    V.validate(raw)
    return V.Criteria("toy", None, "0", overall_pass, criteria, raw)


def _leaf(metric="m", comparator=">", threshold=1, **kw):
    return {"metric": metric, "comparator": comparator, "threshold": threshold, **kw}


def _c(cid, kind, *clauses, **kw):
    return {"id": cid, "source": "src x", "kind": kind, "description": "d",
            "clauses": list(clauses), **kw}  # fmt: skip


def test_expected_criteria_files_exist():
    names = {p.name for p in FILES}
    assert {"f0_validity.yaml", "f0_stuck_scaling.yaml", "f0_headroom.yaml", "a_gate0.yaml",
            "a_success.yaml", "b_claims.yaml", "c_claims.yaml"} <= names  # fmt: skip
    assert (CRITERIA_DIR / "README.md").exists()


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_criteria_file_loads_and_validates(path):
    crit = V.load_criteria(path)
    assert crit.criteria
    for c in crit.criteria:
        assert c["source"].strip()
    # evaluating against an empty frame must not crash and must yield only no-data outcomes
    out = V.evaluate(crit, pd.DataFrame())
    assert out["overall"] in {"PASS", "KILL", "INCONCLUSIVE"}
    assert out["overall"] != "PASS"
    assert all(r["status"] not in {"PASS", "KILL"} for r in out["criteria"])


def test_b_claims_c4_is_deferred():
    crit = V.load_criteria(CRITERIA_DIR / "b_claims.yaml")
    c4 = next(c for c in crit.criteria if c["id"] == "C4")
    assert c4["deferred"] is True
    out = V.evaluate(crit, pd.DataFrame())
    assert next(r for r in out["criteria"] if r["id"] == "C4")["status"] == "DEFERRED"


def test_all_numeric_thresholds_survive_roundtrip():
    # frozen numbers spot-checked against the criteria files
    gate = {c["id"]: c for c in V.load_criteria(CRITERIA_DIR / "a_gate0.yaml").criteria}
    assert gate["G0-a"]["clauses"][0]["threshold"] == 3.0
    assert gate["G0-b"]["clauses"][0]["threshold"] == 0.15
    assert gate["G0-c"]["clauses"][0]["threshold"] == 0.60
    assert gate["G0-d"]["clauses"][0]["threshold"] == 1.0
    assert gate["G0-e"]["clauses"][0]["threshold"] == 0.7


@pytest.mark.skipif(
    not (CRITERIA_DIR / "f0_lowrank.yaml").exists(),
    reason="F0.2 criteria file is not part of this release",
)
def test_f0_lowrank_thresholds_survive_roundtrip():
    k = {c["id"]: c for c in V.load_criteria(CRITERIA_DIR / "f0_lowrank.yaml").criteria}
    assert k["K2.1"]["clauses"][0]["inner"]["threshold"] == 0.50
    assert k["K2.1"]["clauses"][0]["threshold"] == 0.80
    assert k["K2.2"]["clauses"][0]["threshold"] == 1.5


def test_pass_kill_and_inconclusive_semantics():
    df = pd.DataFrame({"m": [0.5, 0.7, 0.9], "grp": ["a", "a", "b"], "flag": [True, False, True]})
    crit = _crit([
        _c("P", "pass", _leaf(aggregation="mean", comparator=">=", threshold=0.7)),
        _c("K", "kill", _leaf(aggregation="max", comparator=">", threshold=5)),
    ])  # fmt: skip
    out = V.evaluate(crit, df)
    st = {r["id"]: r["status"] for r in out["criteria"]}
    assert st == {"P": "PASS", "K": "CLEAR"} and out["overall"] == "PASS"
    kill = _crit([_c("K", "kill", _leaf(aggregation="min", comparator="<", threshold=0.6))])
    assert V.evaluate(kill, df)["overall"] == "KILL"
    notmet = _crit([_c("P", "pass", _leaf(aggregation="max", comparator=">", threshold=2))])
    r = V.evaluate(notmet, df)
    assert r["criteria"][0]["status"] == "NOT_MET" and r["overall"] == "INCONCLUSIVE"
    nodata = _crit([_c("P", "pass", _leaf(metric="absent"))])
    assert V.evaluate(nodata, df)["criteria"][0]["status"] == "INCONCLUSIVE"


def test_filter_frac_group_between_and_query():
    df = pd.DataFrame({"k": [1, 1, 2, 2], "auc": [0.5, 0.55, 0.9, 0.4], "ok": [1, 1, 0, 1]})
    frac = _crit([_c("F", "pass", _leaf("auc", ">=", 0.6, aggregation="frac",
                 inner={"comparator": "<", "threshold": 0.6}, filter={"k": 1}))])  # fmt: skip
    assert V.evaluate(frac, df)["criteria"][0]["status"] == "PASS"  # frac = 1.0
    ge = _crit([_c("F", "pass", _leaf("auc", "==", 1, aggregation="max",
                filter={"auc": {"<": 0.6}, "k": [1, 2]}))])  # fmt: skip
    assert V.evaluate(ge, df)["criteria"][0]["status"] == "NOT_MET"  # max 0.55
    grp = _crit([_c("G", "pass", _leaf("ok", "==", 1, aggregation="min",
                per_group={"by": ["k"], "aggregation": "mean"}))])  # fmt: skip
    # group means: k=1 -> 1.0, k=2 -> 0.5; min over groups = 0.5
    assert V.evaluate(grp, df)["criteria"][0]["detail"]["clauses"][0]["value"] == 0.5
    btw = _crit([_c("B", "pass", _leaf("auc", "between", [0.5, 0.6], aggregation="median",
                filter={"k": 1}))])  # fmt: skip
    assert V.evaluate(btw, df)["criteria"][0]["status"] == "PASS"
    q = _crit([_c("Q", "pass", _leaf("auc", ">=", 1, aggregation="count",
              filter={"query": "auc > 0.5 and ok == 1"}))])  # fmt: skip
    assert V.evaluate(q, df)["criteria"][0]["status"] == "PASS"


def test_nested_groups_refs_scopes_and_special_kinds():
    df = pd.DataFrame({"a": [1.0], "b": [5.0]})
    crit = _crit([
        _c("A", "kill", _leaf("a", "<", 0), scope="s1"),
        _c("B", "kill", {"ref": "A"}, _leaf("b", ">", 1), combine="any", scope="s2"),
        _c("C", "pass", {"combine": "any", "clauses": [_leaf("a", ">", 9), _leaf("b", ">", 1)]},
           scope="s1"),
        _c("D", "report", _leaf("a", ">", 0)),
        _c("E", "redirect", _leaf("b", ">", 1)),
        {"id": "M", "source": "p", "kind": "pass", "description": "d", "manual": True},
        _c("X", "pass", _leaf("a", ">", 0), deferred=True),
    ])  # fmt: skip
    out = V.evaluate(crit, df)
    st = {r["id"]: r["status"] for r in out["criteria"]}
    assert st == {"A": "CLEAR", "B": "KILL", "C": "PASS", "D": "REPORTED", "E": "REDIRECT",
                  "M": "INCONCLUSIVE", "X": "DEFERRED"}  # fmt: skip
    assert out["overall"] == "KILL"
    assert out["by_scope"] == {"s1": "PASS", "s2": "KILL"}


def test_overall_pass_any_vs_all():
    df = pd.DataFrame({"a": [1.0]})
    cl = [_c("P1", "pass", _leaf("a", ">", 0)), _c("P2", "pass", _leaf("a", ">", 5))]
    assert V.evaluate(_crit(cl, "all"), df)["overall"] == "INCONCLUSIVE"
    assert V.evaluate(_crit(cl, "any"), df)["overall"] == "PASS"


def test_schema_rejects_bad_files():
    good = _c("A", "pass", _leaf())

    def bad(**changes):
        raw = {"schema_version": 1, "id": "t", "title": "t", "source_plan": "p", "frozen": "x",
               "criteria": [good]}  # fmt: skip
        raw.update(changes)
        with pytest.raises(V.CriteriaError):
            V.validate(raw)

    bad(schema_version=2)
    bad(unknown=1)
    bad(criteria=[good, good])  # duplicate id
    bad(criteria=[{**good, "source": ""}])
    bad(criteria=[{**good, "kind": "maybe"}])
    bad(criteria=[_c("A", "pass", _leaf(comparator="~"))])
    bad(criteria=[_c("A", "pass", _leaf(comparator="between", threshold=1))])
    bad(criteria=[_c("A", "pass", _leaf(aggregation="frac"))])  # frac needs inner
    bad(criteria=[_c("A", "pass", {"ref": "nope"})])
    bad(criteria=[_c("A", "pass", {"ref": "B"}), _c("B", "pass", {"ref": "A"})])  # cycle
    bad(criteria=[_c("A", "pass", {**_leaf(), "typo": 1})])


def test_write_verdict_and_cli(tmp_path):
    crit_path = tmp_path / "c.yaml"
    crit_path.write_text(yaml.safe_dump({
        "schema_version": 1, "id": "toy", "title": "t", "source_plan": "p", "frozen": "x",
        "criteria": [_c("P", "pass", _leaf("m", ">", 0))],
    }))  # fmt: skip
    csv = tmp_path / "metrics.csv"
    pd.DataFrame({"m": [1.0, 2.0]}).to_csv(csv, index=False)
    assert V.main([str(crit_path), str(csv)]) == 0
    data = json.loads((tmp_path / "verdict.json").read_text())
    assert data["overall"] == "PASS" and len(data["criteria_sha256"]) == 64


# --------------------------------------------------------------------------- harness


def _fake_git(dirty: bool):
    return lambda: {"commit": "abc123", "branch": "b", "dirty": dirty}


def test_write_run_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "capture_git_info", _fake_git(False))
    df = pd.DataFrame({"x": [1, 2], "y": [0.5, 0.25]})
    cfg = {"exp": "toy", "tile": 64}
    run = harness.write_run(tmp_path / "toy", cfg, 7, df, {"p": {"value": 1}}, {"note": "n"})
    assert run.parent == tmp_path / "toy"
    for name in ("config.json", "run_metadata.json", "metrics.csv", "params.json"):
        assert (run / name).exists()
    meta = json.loads((run / "run_metadata.json").read_text())
    assert meta["git"]["commit"] == "abc123" and meta["git"]["dirty"] is False
    assert meta["seed"] == 7 and meta["allow_dirty"] is False and meta["extra"] == {"note": "n"}
    assert "torch_version" in meta["environment"]
    assert pd.read_csv(run / "metrics.csv").equals(df)
    assert json.loads((run / "config.json").read_text()) == cfg


def test_write_run_refuses_dirty_tree_unless_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "capture_git_info", _fake_git(True))
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(harness.DirtyTreeError):
        harness.write_run(tmp_path, {}, 0, df, {})
    assert not any(tmp_path.iterdir())  # nothing written on refusal
    run = harness.write_run(tmp_path, {}, 0, df, {}, allow_dirty=True)
    assert json.loads((run / "run_metadata.json").read_text())["allow_dirty"] is True


def _square(point: dict) -> dict:
    return {"sq": point["x"] ** 2}


def test_run_grid_order_resume_and_parallel(tmp_path):
    grid = [{"x": i, "tag": "a"} for i in range(5)]
    df = harness.run_grid(_square, grid, jobs=1, out_dir=tmp_path)
    assert list(df["sq"]) == [0, 1, 4, 9, 16] and list(df["x"]) == list(range(5))
    calls = []

    def boom(p):  # would fail if any cell were recomputed
        calls.append(p)
        raise AssertionError

    again = harness.run_grid(boom, grid, jobs=1, out_dir=tmp_path)
    assert again.equals(df) and not calls
    par = harness.run_grid(_square, grid, jobs=2)
    assert list(par["sq"]) == [0, 1, 4, 9, 16]


def test_scenario_as_dict_is_json_serializable():
    from neurofault.spectral.types import EncodingSpec, ProgNoise, ReadNoise, TileSpec

    s = harness.Scenario(TileSpec(-1, "syn", 0, 0, 8, 8, 8), EncodingSpec(), ProgNoise(),
                         ReadNoise(), None, None, 20.0, 1)  # fmt: skip
    d = s.as_dict()
    json.dumps(d)
    assert d["tile"]["__type__"] == "TileSpec" and d["enc"]["kind"] == "diff_pair"
