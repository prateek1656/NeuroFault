"""Mechanical pass/kill verdicts from pre-registered criteria.

Why: thresholds are frozen in experiments/spectral/criteria/*.yaml BEFORE the first full run;
this module is the only thing that turns a metrics table into PASS / KILL / INCONCLUSIVE, so
nobody (including us) judges a number by eye after seeing it.

YAML SCHEMA (schema_version: 1). Criteria files are trusted repo code (`filter.query` is passed
to pandas DataFrame.query).

  schema_version: 1
  id: <file id>                    title: <text>
  source_plan: <registered source>   frozen: <YYYY-MM-DD>
  overall_pass: all | any          # how pass-kind criteria combine into the overall PASS
  note: <optional text>
  criteria:
    - id: H1.1                     # unique in file
      source: "F0.1 H1.1"                   # REQUIRED criterion citation
      kind: pass | kill | escalate | redirect | report
      scope: <optional claim tag>  # by_scope sub-verdicts in verdict.json
      description: <verbatim-ish plan text>
      note: <optional: ambiguity flag or interpretation>
      deferred: true               # frozen but not evaluated (status DEFERRED)
      manual: true                 # no clauses; needs human judgement (INCONCLUSIVE)
      combine: all | any           # default all; how `clauses` combine
      clauses:                     # leaves and/or nested {combine, clauses} groups
        - metric: <column>         # leaf: aggregate this column of the metrics DataFrame
          aggregation: mean|median|min|max|sum|count|nunique|std|quantile|frac   (default mean)
          q: 0.9                   # quantile only
          inner: {comparator: "<", threshold: 0.6}   # frac only: fraction of rows satisfying it
          filter: {col: value | [values] | {">=": 128}, query: "pandas expr"}   # rows kept
          per_group: {by: [cols], aggregation: median}  # aggregate within groups first
          comparator: "<"|"<="|">"|">="|"=="|"!="|between|outside
          threshold: number | [lo, hi]     # [lo, hi] inclusive for between/outside
          min_n: 1                 # rows (or groups) required, else the clause has no data
        - ref: <criterion id>      # truth of another criterion's condition (not its status)
        - combine: any             # nested group
          clauses: [ ... ]

  Boolean columns aggregate as 0/1, so mean(bool) is a fraction. Derived columns (ratios,
  slopes, per-row flags) are produced by the experiment script; the metric name is the contract.

SEMANTICS. A clause is True / False / None (None = metric or rows missing -> no data). Groups use
Kleene logic. Per criterion status:
  pass:     True PASS | False NOT_MET | None INCONCLUSIVE
  kill:     True KILL | False CLEAR   | None INCONCLUSIVE
  escalate: True ESCALATE; redirect: True REDIRECT (human decision; else CLEAR / INCONCLUSIVE)
  report:   REPORTED (condition recorded, never affects the overall verdict)
  deferred: DEFERRED (ignored).  manual: INCONCLUSIVE.
Overall: KILL if any kill criterion fired; else INCONCLUSIVE if any ESCALATE/REDIRECT fired or any
kill criterion lacks data; else PASS if the pass criteria satisfy overall_pass (all|any) and at
least one exists; else INCONCLUSIVE. "Pass not met, nothing killed" is the registered intermediate
outcome and stays INCONCLUSIVE. `by_scope` repeats the same rule per scope tag: multi-claim files
(B, C') should be read through it, since a scoped kill (e.g. drop C2) is not a global kill.

CLI: python -m neurofault.spectral.verdict <criteria.yaml> <metrics.csv> [--out verdict.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

KINDS = {"pass", "kill", "escalate", "redirect", "report"}
AGGS = {"mean", "median", "min", "max", "sum", "count", "nunique", "std", "quantile", "frac"}
CMPS = {"<", "<=", ">", ">=", "==", "!=", "between", "outside"}
_OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}
_CRIT_KEYS = {
    "id", "source", "kind", "scope", "description", "note", "deferred", "manual", "combine",
    "clauses",
}  # fmt: skip
_LEAF_KEYS = {
    "metric", "aggregation", "q", "inner", "filter", "per_group", "comparator", "threshold",
    "min_n", "label", "note",
}  # fmt: skip
_FILE_KEYS = {
    "schema_version", "id", "title", "source_plan", "frozen", "overall_pass", "note", "criteria",
}  # fmt: skip


class CriteriaError(ValueError):
    """The criteria file violates the schema."""


@dataclass
class Criteria:
    id: str
    path: Path | None
    sha256: str
    overall_pass: str
    criteria: list[dict]
    raw: dict = field(repr=False, default_factory=dict)


# --------------------------------------------------------------------------- loading


def _check_cmp(c, thr, where: str) -> None:
    if c not in CMPS:
        raise CriteriaError(f"{where}: bad comparator {c!r}")
    if c in ("between", "outside"):
        if not (isinstance(thr, list) and len(thr) == 2 and thr[0] <= thr[1]):
            raise CriteriaError(f"{where}: {c} needs threshold [lo, hi]")
    elif isinstance(thr, bool) or not isinstance(thr, (int, float)):
        raise CriteriaError(f"{where}: threshold must be a number, got {thr!r}")


def _validate_node(node: dict, where: str, ids: set[str]) -> None:
    if not isinstance(node, dict):
        raise CriteriaError(f"{where}: clause must be a mapping")
    if "ref" in node:
        if set(node) != {"ref"}:
            raise CriteriaError(f"{where}: 'ref' clause takes no other keys")
        if node["ref"] not in ids:
            raise CriteriaError(f"{where}: unknown ref {node['ref']!r}")
        return
    if "clauses" in node:
        if set(node) - {"combine", "clauses"}:
            raise CriteriaError(f"{where}: group takes only combine/clauses")
        if node.get("combine", "all") not in ("all", "any"):
            raise CriteriaError(f"{where}: combine must be all|any")
        if not node["clauses"]:
            raise CriteriaError(f"{where}: empty group")
        for i, c in enumerate(node["clauses"]):
            _validate_node(c, f"{where}.clauses[{i}]", ids)
        return
    extra = set(node) - _LEAF_KEYS
    if extra:
        raise CriteriaError(f"{where}: unknown leaf keys {sorted(extra)}")
    for req in ("metric", "comparator", "threshold"):
        if req not in node:
            raise CriteriaError(f"{where}: leaf missing {req!r}")
    agg = node.get("aggregation", "mean")
    if agg not in AGGS:
        raise CriteriaError(f"{where}: bad aggregation {agg!r}")
    if agg == "frac":
        inner = node.get("inner")
        if not isinstance(inner, dict):
            raise CriteriaError(f"{where}: frac needs inner {{comparator, threshold}}")
        _check_cmp(inner.get("comparator"), inner.get("threshold"), f"{where}.inner")
    if agg == "quantile" and not 0.0 <= node.get("q", -1) <= 1.0:
        raise CriteriaError(f"{where}: quantile needs q in [0, 1]")
    pg = node.get("per_group")
    if pg is not None:
        if not isinstance(pg.get("by"), list) or pg.get("aggregation", "mean") not in AGGS:
            raise CriteriaError(f"{where}: per_group needs by: [cols] and a valid aggregation")
        if pg.get("aggregation") == "frac":
            raise CriteriaError(f"{where}: per_group aggregation 'frac' unsupported; use a bool")
    _check_cmp(node["comparator"], node["threshold"], where)


def _refs(node: dict) -> list[str]:
    if "ref" in node:
        return [node["ref"]]
    return [r for c in node.get("clauses", []) for r in _refs(c)]


def validate(raw: dict) -> None:
    if not isinstance(raw, dict):
        raise CriteriaError("criteria file must be a mapping")
    extra = set(raw) - _FILE_KEYS
    if extra:
        raise CriteriaError(f"unknown top-level keys {sorted(extra)}")
    if raw.get("schema_version") != 1:
        raise CriteriaError("schema_version must be 1")
    for req in ("id", "title", "source_plan", "frozen", "criteria"):
        if req not in raw:
            raise CriteriaError(f"missing top-level {req!r}")
    if raw.get("overall_pass", "all") not in ("all", "any"):
        raise CriteriaError("overall_pass must be all|any")
    crits = raw["criteria"]
    if not isinstance(crits, list) or not crits:
        raise CriteriaError("criteria must be a non-empty list")
    ids = [c.get("id") for c in crits]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) and i for i in ids):
        raise CriteriaError("criterion ids must be unique non-empty strings")
    idset = set(ids)
    for c in crits:
        w = f"criterion {c['id']}"
        extra = set(c) - _CRIT_KEYS
        if extra:
            raise CriteriaError(f"{w}: unknown keys {sorted(extra)}")
        if not c.get("source"):
            raise CriteriaError(f"{w}: 'source' (plan + section) is required")
        if c.get("kind") not in KINDS:
            raise CriteriaError(f"{w}: kind must be one of {sorted(KINDS)}")
        if not c.get("description"):
            raise CriteriaError(f"{w}: description required")
        if c.get("manual"):
            if c.get("clauses"):
                raise CriteriaError(f"{w}: manual criteria carry no clauses")
            continue
        if not c.get("clauses"):
            raise CriteriaError(f"{w}: needs clauses (or manual: true)")
        _validate_node({"combine": c.get("combine", "all"), "clauses": c["clauses"]}, w, idset)
    graph = {c["id"]: {r for cl in c.get("clauses", []) for r in _refs(cl)} for c in crits}
    state: dict[str, int] = {}

    def visit(n: str) -> None:
        if state.get(n) == 1:
            raise CriteriaError(f"ref cycle through {n!r}")
        if state.get(n) == 2:
            return
        state[n] = 1
        for m in graph[n]:
            visit(m)
        state[n] = 2

    for n in graph:
        visit(n)


def load_criteria(path: str | Path) -> Criteria:
    p = Path(path)
    text = p.read_bytes()
    raw = yaml.safe_load(text)
    validate(raw)
    return Criteria(
        id=raw["id"],
        path=p,
        sha256=hashlib.sha256(text).hexdigest(),
        overall_pass=raw.get("overall_pass", "all"),
        criteria=raw["criteria"],
        raw=raw,
    )


# --------------------------------------------------------------------------- evaluation


def _cmp(value: float, comparator: str, thr) -> bool:
    if comparator == "between":
        return bool(thr[0] <= value <= thr[1])
    if comparator == "outside":
        return bool(value < thr[0] or value > thr[1])
    return bool(_OPS[comparator](value, thr))


def _filter(df: pd.DataFrame, flt: dict | None) -> pd.DataFrame:
    if not flt:
        return df
    out = df
    for k, v in flt.items():
        if k == "query":
            out = out.query(v)
            continue
        if k not in out.columns:
            raise KeyError(k)
        if isinstance(v, dict):
            for op, val in v.items():
                if op not in _OPS:
                    raise CriteriaError(f"filter op {op!r}")
                out = out[_OPS[op](out[k], val)]
        elif isinstance(v, list):
            out = out[out[k].isin(v)]
        else:
            out = out[out[k] == v]
    return out


def _aggregate(s: pd.Series, agg: str, q: float | None, inner: dict | None) -> float:
    if agg == "count":
        return float(len(s))
    if agg == "nunique":
        return float(s.nunique())
    if agg == "frac":
        return float(np.mean([_cmp(v, inner["comparator"], inner["threshold"]) for v in s]))
    if agg == "quantile":
        return float(s.quantile(q))
    return float(getattr(s, agg)())


def _eval_leaf(node: dict, df: pd.DataFrame) -> dict:
    res = {"metric": node["metric"], "aggregation": node.get("aggregation", "mean"),
           "comparator": node["comparator"], "threshold": node["threshold"],
           "value": None, "n": 0, "result": None, "reason": None}  # fmt: skip
    try:
        d = _filter(df, node.get("filter"))
    except Exception as e:  # noqa: BLE001 - missing column or bad query = no data
        res["reason"] = f"filter failed: {e!r}"
        return res
    if node["metric"] not in d.columns:
        res["reason"] = f"metric column {node['metric']!r} missing"
        return res
    agg, q, inner = node.get("aggregation", "mean"), node.get("q"), node.get("inner")
    min_n = node.get("min_n", 1)
    pg = node.get("per_group")
    if pg:
        missing = [c for c in pg["by"] if c not in d.columns]
        if missing:
            res["reason"] = f"group column(s) {missing} missing"
            return res
        sub = d[[node["metric"], *pg["by"]]].copy()
        sub[node["metric"]] = pd.to_numeric(sub[node["metric"]].astype(float), errors="coerce")
        sub = sub.dropna(subset=[node["metric"]])
        groups = [
            _aggregate(g[node["metric"]], pg.get("aggregation", "mean"), pg.get("q"), None)
            for _, g in sub.groupby(pg["by"], sort=True)
        ]
        res["n"] = len(groups)
        series = pd.Series(groups, dtype=float)
    else:
        series = pd.to_numeric(d[node["metric"]].astype(float), errors="coerce").dropna()
        res["n"] = len(series)
    if res["n"] < min_n:
        res["reason"] = f"only {res['n']} rows/groups after filter (min_n={min_n})"
        return res
    value = _aggregate(series, agg, q, inner)
    res["value"] = value
    res["result"] = _cmp(value, node["comparator"], node["threshold"])
    return res


def _kleene(vals: list[bool | None], combine: str) -> bool | None:
    if combine == "all":
        if any(v is False for v in vals):
            return False
        return None if any(v is None for v in vals) else True
    if any(v is True for v in vals):
        return True
    return None if any(v is None for v in vals) else False


def _eval_node(node: dict, df: pd.DataFrame, crits: dict, memo: dict) -> tuple[bool | None, dict]:
    if "ref" in node:
        val = _cond(node["ref"], df, crits, memo)
        return val, {"ref": node["ref"], "result": val}
    if "clauses" in node:
        kids = [_eval_node(c, df, crits, memo) for c in node["clauses"]]
        comb = node.get("combine", "all")
        return _kleene([k[0] for k in kids], comb), {
            "combine": comb,
            "result": _kleene([k[0] for k in kids], comb),
            "clauses": [k[1] for k in kids],
        }
    leaf = _eval_leaf(node, df)
    return leaf["result"], leaf


def _cond(cid: str, df: pd.DataFrame, crits: dict, memo: dict) -> bool | None:
    if cid not in memo:
        c = crits[cid]
        if c.get("manual") or c.get("deferred"):
            memo[cid] = (None, {})
        else:
            memo[cid] = _eval_node(
                {"combine": c.get("combine", "all"), "clauses": c["clauses"]}, df, crits, memo
            )
    return memo[cid][0]


_STATUS = {
    "pass": {True: "PASS", False: "NOT_MET", None: "INCONCLUSIVE"},
    "kill": {True: "KILL", False: "CLEAR", None: "INCONCLUSIVE"},
    "escalate": {True: "ESCALATE", False: "CLEAR", None: "INCONCLUSIVE"},
    "redirect": {True: "REDIRECT", False: "CLEAR", None: "INCONCLUSIVE"},
}


def _overall(results: list[dict], overall_pass: str) -> str:
    live = [r for r in results if r["status"] != "DEFERRED" and r["kind"] != "report"]
    if any(r["status"] == "KILL" for r in live):
        return "KILL"
    if any(r["status"] in ("ESCALATE", "REDIRECT") for r in live):
        return "INCONCLUSIVE"
    if any(r["kind"] == "kill" and r["status"] == "INCONCLUSIVE" for r in live):
        return "INCONCLUSIVE"
    passes = [r for r in live if r["kind"] == "pass"]
    if not passes:
        return "INCONCLUSIVE"
    ok = [r["status"] == "PASS" for r in passes]
    return "PASS" if (all(ok) if overall_pass == "all" else any(ok)) else "INCONCLUSIVE"


def evaluate(criteria: Criteria, metrics: pd.DataFrame) -> dict:
    """Evaluate every criterion against `metrics`; return the verdict dict (see module doc)."""
    crits = {c["id"]: c for c in criteria.criteria}
    memo: dict = {}
    results = []
    for c in criteria.criteria:
        base = {"id": c["id"], "kind": c["kind"], "scope": c.get("scope"),
                "source": c["source"], "description": c["description"],
                "note": c.get("note")}  # fmt: skip
        if c.get("deferred"):
            results.append({**base, "status": "DEFERRED", "condition": None, "detail": {}})
            continue
        if c.get("manual"):
            st = "REPORTED" if c["kind"] == "report" else "INCONCLUSIVE"
            results.append({**base, "status": st, "condition": None,
                            "detail": {"reason": "requires human judgement"}})  # fmt: skip
            continue
        _cond(c["id"], metrics, crits, memo)
        cond, detail = memo[c["id"]]
        status = "REPORTED" if c["kind"] == "report" else _STATUS[c["kind"]][cond]
        results.append({**base, "status": status, "condition": cond, "detail": detail})
    scopes = sorted({r["scope"] for r in results if r["scope"]})
    return {
        "criteria_id": criteria.id,
        "criteria_sha256": criteria.sha256,
        "overall": _overall(results, criteria.overall_pass),
        "overall_pass_rule": criteria.overall_pass,
        "by_scope": {
            s: _overall([r for r in results if r["scope"] == s], criteria.overall_pass)
            for s in scopes
        },
        "n_rows": len(metrics),
        "criteria": results,
    }


def write_verdict(verdict: dict, path: str | Path, *, extra: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**verdict, **(extra or {})}, indent=2, default=str))
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("criteria")
    ap.add_argument("metrics_csv")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    v = evaluate(load_criteria(a.criteria), pd.read_csv(a.metrics_csv))
    out = a.out or str(Path(a.metrics_csv).with_name("verdict.json"))
    write_verdict(v, out, extra={"metrics_csv": a.metrics_csv})
    print(f"{v['criteria_id']}: {v['overall']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
