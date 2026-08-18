#!/usr/bin/env python3
"""SWO69 T5 — correlation / regression / figure generation (SWO_PLAN.md §6).

Inputs (produced by T1/T3 against the same run dir):
  <run-dir>/kg_quality/kg_quality_per_sample.csv
  <run-dir>/discovery_metrics/discovery_metrics_per_sample.csv
  [E2] --e2-dir: parent dir of per-perturbation run dirs, each named
       <system>_<dataset>_<perturbation> (see T6 run_experiment_swo69.sh E2 mode)

Outputs (default <run-dir>/analysis/):
  table3_spearman.csv      — quality × performance Spearman r_s with p (Table 3)
  table3_significant.csv   — pairs with p < 0.05
  regression_fixed_effects.csv — F1 ~ ER + RR + Density + System + Dataset (OLS)
  table2_by_system.csv     — merged per-system quality/performance summary (Table 2)
  bootstrap_ci.csv         — 95% CIs (1000 resamples) for key statistics
  wilcoxon_e2.csv          — P0 vs Pk paired tests (E2 mode)
  fig1_er_vs_f1.png        — Figure 1
  fig2_degradation.png     — Figure 2 (E2 mode)
  fig3_heatmap.png         — Figure 3

No statsmodels/sklearn: OLS is numpy lstsq; inference via scipy.stats.

Usage:
  python3 scripts/correlation_analysis.py --run-dir eval_results/<e0_run>
  python3 scripts/correlation_analysis.py --e2-dir eval_results/swo69_e2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

QUALITY_METRICS = ["er", "rr", "density", "kgc"]
PERF_METRICS = ["rh_at_5", "cic", "support_recall_at_5", "f1", "faithfulness"]

PERT_ORDER = ["p0", "p1_10", "p1_30", "p1_50", "p2_10", "p2_30", "p3", "p5_10", "p5_30"]
PERT_LABEL = {
    "p0": "P0",
    "p1_10": "P1 10%",
    "p1_30": "P1 30%",
    "p1_50": "P1 50%",
    "p2_10": "P2 10%",
    "p2_30": "P2 30%",
    "p3": "P3",
    "p4": "P4",
    "p5_10": "P5 10%",
    "p5_30": "P5 30%",
}


def fnum(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sig_stars(p: float | None) -> str:
    if p is None:
        return ""
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# --------------------------------------------------------------------------- #
# Table 3 — Spearman
# --------------------------------------------------------------------------- #


def spearman_matrix(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    cells = []
    for q in QUALITY_METRICS:
        for p in PERF_METRICS:
            xs, ys = [], []
            for r in rows:
                x, y = fnum(r.get(q)), fnum(r.get(p))
                if x is not None and y is not None:
                    xs.append(x)
                    ys.append(y)
            n = len(xs)
            if n < 4 or (len(set(xs)) < 2 or len(set(ys)) < 2):
                cells.append({"quality": q, "performance": p, "n": n, "rs": None, "p": None})
                continue
            res = stats.spearmanr(xs, ys)
            rs = float(res.statistic) if res.statistic is not None else None
            pv = float(res.pvalue) if res.pvalue is not None else None
            cells.append({"quality": q, "performance": p, "n": n, "rs": rs, "p": pv})
    significant = [c for c in cells if c["p"] is not None and c["p"] < 0.05]
    return cells, significant


def bootstrap_ci(rows: list[dict], metric_x: str, metric_y: str, iters: int = 1000, seed: int = 42) -> tuple | None:
    xs = [fnum(r.get(metric_x)) for r in rows]
    ys = [fnum(r.get(metric_y)) for r in rows]
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 4:
        return None
    rng = np.random.default_rng(seed)
    n = len(pairs)
    rs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        a = np.array([pairs[j][0] for j in idx], dtype=float)
        b = np.array([pairs[j][1] for j in idx], dtype=float)
        if len(set(a)) < 2 or len(set(b)) < 2:
            rs[i] = np.nan
            continue
        r, _ = stats.spearmanr(a, b)
        rs[i] = r
    rs = rs[~np.isnan(rs)]
    if len(rs) < 10:
        return None
    return (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))


# --------------------------------------------------------------------------- #
# Fixed-effects regression (OLS via numpy)
# --------------------------------------------------------------------------- #


def ols_fixed_effects(rows: list[dict]) -> list[dict]:
    """F1 ~ ER + RR + Density + System dummies + Dataset dummies."""
    idx = []
    for i, r in enumerate(rows):
        f1 = fnum(r.get("f1"))
        if f1 is None:
            continue
        if any(fnum(r.get(m)) is None for m in ("er", "rr", "density")):
            continue
        idx.append(i)
    if len(idx) < 8:
        return []
    ys = np.array([fnum(rows[i]["f1"]) for i in idx], dtype=float)
    Xs = [
        np.array([fnum(rows[i]["er"]) for i in idx], dtype=float),
        np.array([fnum(rows[i]["rr"]) for i in idx], dtype=float),
        np.array([fnum(rows[i]["density"]) for i in idx], dtype=float),
    ]
    names = ["ER", "RR", "Density"]
    systems = sorted({rows[i].get("system") or rows[i].get("condition") or "?" for i in idx})
    datasets = sorted({rows[i].get("dataset") or "?" for i in idx})
    for s in systems[1:]:
        names.append(f"System={s}")
        Xs.append(np.array([1.0 if (rows[i].get("system") or rows[i].get("condition")) == s else 0.0 for i in idx]))
    for d in datasets[1:]:
        names.append(f"Dataset={d}")
        Xs.append(np.array([1.0 if rows[i].get("dataset") == d else 0.0 for i in idx]))
    X = np.column_stack([np.ones(len(idx))] + Xs)
    names = ["intercept"] + names
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
    resid = ys - X @ beta
    dof = n - k
    if dof <= 0:
        return []
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tvals = np.where(se > 0, beta / np.where(se > 0, se, 1.0), 0.0)
    pvals = np.array([2.0 * stats.t.sf(abs(t), dof) for t in tvals])
    y_sd = float(ys.std(ddof=1))
    out = []
    for j, name in enumerate(names):
        x_sd = float(X[:, j].std(ddof=1)) if j > 0 else None
        std_beta = (beta[j] * x_sd / y_sd) if (x_sd and x_sd > 0) else None
        out.append(
            {
                "term": name,
                "coef": float(beta[j]),
                "se": float(se[j]),
                "t": float(tvals[j]),
                "p": float(pvals[j]),
                "std_beta": float(std_beta) if std_beta is not None else None,
            }
        )
    r2 = 1.0 - float(resid @ resid) / float(((ys - ys.mean()) ** 2).sum())
    for row in out:
        row["r2"] = r2
        row["n"] = n
    return out


# --------------------------------------------------------------------------- #
# E2 — degradation + Wilcoxon
# --------------------------------------------------------------------------- #


def parse_e2_name(dirname: str) -> tuple[str, str, str] | None:
    m = re.fullmatch(r"([a-z]+)_(musique|causal|webis)_(p\d+(_\d+)?)", dirname)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = re.fullmatch(r"([a-z]+)_(musique|causal|webis)_([a-z]+)_(p\d+(_\d+)?)", dirname)
    if m:
        return m.group(1), m.group(2), m.group(4)
    return None


def run_e2(e2_dir: Path, out_dir: Path) -> int:
    rows = []
    for d in sorted(e2_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_e2_name(d.name)
        if parsed is None:
            continue
        system, dataset, pert = parsed
        dm = read_csv(d / "discovery_metrics" / "discovery_metrics_by_condition.csv")
        for cond_row in dm:
            rows.append(
                {
                    "system": system,
                    "dataset": dataset,
                    "perturbation": pert,
                    "condition": cond_row.get("condition"),
                    "n_samples": fnum(cond_row.get("n_samples")),
                    "rh_at_5": fnum(cond_row.get("rh_at_5")),
                    "f1": fnum(cond_row.get("f1")),
                    "kgc": fnum(cond_row.get("kgc")),
                }
            )
    if not rows:
        print(f"[e2] no E2 runs found under {e2_dir} (expected <system>_<dataset>_<pert> dirs)", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    # Wilcoxon: P0 vs each other perturbation, per (system, dataset)
    wil = []
    groups = {}
    for r in rows:
        groups.setdefault((r["system"], r["dataset"]), {})[r["perturbation"]] = r
    for (system, dataset), per in sorted(groups.items()):
        base = per.get("p0")
        if base is None:
            continue
        for pert, r in sorted(per.items()):
            if pert == "p0":
                continue
            for metric in ("f1", "rh_at_5"):
                a, b = base.get(metric), r.get(metric)
                if a is None or b is None:
                    continue
                diff = a - b
                if diff == 0:
                    wil.append({"system": system, "dataset": dataset, "perturbation": pert, "metric": metric, "n": None, "stat": None, "p": None, "note": "no difference"})
                    continue
                # Without per-sample vectors we can only test the aggregated
                # values when a per-sample file is present.
                per_sample = read_csv(e2_dir / f"{system}_{dataset}_{pert}" / "discovery_metrics" / "discovery_metrics_per_sample.csv")
                per_base = read_csv(e2_dir / f"{system}_{dataset}_p0" / "discovery_metrics" / "discovery_metrics_per_sample.csv")
                if len(per_sample) == len(per_base) and per_sample:
                    va = np.array([fnum(r0.get(metric)) for r0 in per_base], dtype=float)
                    vb = np.array([fnum(r0.get(metric)) for r0 in per_sample], dtype=float)
                    ok = ~(np.isnan(va) | np.isnan(vb))
                    va, vb = va[ok], vb[ok]
                    if len(va) >= 3:
                        res = stats.wilcoxon(va, vb)
                        wil.append(
                            {"system": system, "dataset": dataset, "perturbation": pert, "metric": metric,
                             "n": int(len(va)), "stat": float(res.statistic), "p": float(res.pvalue), "note": ""}
                        )
                        continue
                wil.append({"system": system, "dataset": dataset, "perturbation": pert, "metric": metric,
                            "n": None, "stat": None, "p": None, "note": "insufficient paired data"})
    with (out_dir / "wilcoxon_e2.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["system", "dataset", "perturbation", "metric", "n", "stat", "p", "note"])
        w.writeheader()
        for row in wil:
            w.writerow(row)

    # Figure 2: degradation curves (P1/P2/P5 series; P3 plotted at its own slot)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for system in sorted({r["system"] for r in rows}):
        sub = [r for r in rows if r["system"] == system]
        for ax, metric in ((axes[0], "rh_at_5"), (axes[1], "f1")):
            pts = [
                (PERT_ORDER.index(r["perturbation"]), r[metric])
                for r in sub
                if r["perturbation"] in PERT_ORDER and r[metric] is not None
            ]
            pts.sort(key=lambda p: p[0])
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=5, lw=1.2, label=system)
    axes[0].set_ylabel("Retrieval-Hit@5")
    axes[1].set_ylabel("Answer F1")
    axes[0].set_title("Figure 2a — RH@5 under KG perturbation")
    axes[1].set_title("Figure 2b — F1 under KG perturbation")
    tickpos = [PERT_ORDER.index(p) for p in PERT_ORDER if any(r["perturbation"] == p for r in rows)]
    ticks = [PERT_LABEL[p] for p in (PERT_ORDER[i] for i in tickpos)]
    for ax in axes:
        ax.set_xticks(tickpos)
        ax.set_xticklabels(ticks, rotation=30, ha="right")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_degradation.png", dpi=150)
    plt.close(fig)
    print(f"[e2] wrote {out_dir/'wilcoxon_e2.csv'} and {out_dir/'fig2_degradation.png'} ({len(rows)} condition rows)")
    return 0


# --------------------------------------------------------------------------- #
# Main (E0/E1 analysis)
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--e2-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.e2_dir:
        out_dir = args.out or (args.e2_dir / "analysis")
        return run_e2(args.e2_dir.resolve(), out_dir.resolve())

    if not args.run_dir:
        print("error: provide --run-dir (E0/E1) or --e2-dir (E2)", file=sys.stderr)
        return 2
    run_dir: Path = args.run_dir
    out_dir = (args.out or (run_dir / "analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kg_rows = read_csv(run_dir / "kg_quality" / "kg_quality_per_sample.csv")
    dm_rows = read_csv(run_dir / "discovery_metrics" / "discovery_metrics_per_sample.csv")
    if not kg_rows and not dm_rows:
        print(f"error: no kg_quality/discovery_metrics CSVs under {run_dir}", file=sys.stderr)
        return 2

    # Merge on (system|condition, sample_id)
    dm_by_key = {}
    for r in dm_rows:
        key = (r.get("system") or r.get("condition"), r.get("sample_id"))
        dm_by_key.setdefault(key, r)
    merged = []
    for r in kg_rows:
        key = (r.get("system"), r.get("sample_id"))
        d = dm_by_key.get(key, {})
        m = dict(r)
        for col in PERF_METRICS + ["kgc", "cig", "cic", "cic_all", "plr", "exact_match", "support_recall_at_5", "bridge_coverage_at_5", "mrr_at_5", "ndcg_at_5", "faithfulness", "error"]:
            if col not in m or m[col] in (None, ""):
                m[col] = d.get(col)
        m["system"] = r.get("system")
        m["condition"] = d.get("condition", r.get("system"))
        m["dataset"] = d.get("dataset") or r.get("dataset")
        merged.append(m)
    if not merged:
        print("error: could not merge any rows (system/sample_id mismatch)", file=sys.stderr)
        return 2

    # Table 3
    cells, significant = spearman_matrix(merged)
    with (out_dir / "table3_spearman.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quality", "performance", "n", "rs", "p", "stars"])
        w.writeheader()
        for c in cells:
            row = dict(c)
            row["stars"] = sig_stars(c["p"])
            w.writerow(row)
    with (out_dir / "table3_significant.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quality", "performance", "n", "rs", "p", "stars"])
        w.writeheader()
        for c in sorted(significant, key=lambda x: x["p"] or 1.0):
            row = dict(c)
            row["stars"] = sig_stars(c["p"])
            w.writerow(row)

    # Regression
    reg = ols_fixed_effects(merged)
    with (out_dir / "regression_fixed_effects.csv").open("w", newline="", encoding="utf-8") as fh:
        if reg:
            w = csv.DictWriter(fh, fieldnames=["term", "coef", "se", "t", "p", "std_beta", "n", "r2"])
            w.writeheader()
            for row in reg:
                w.writerow(row)
        else:
            fh.write("insufficient rows for regression\n")

    # Bootstrap CIs for the key correlations (Table 3 headline)
    boot_rows = []
    for q in ("er", "kgc"):
        for p in ("f1", "rh_at_5"):
            ci = bootstrap_ci(merged, q, p, iters=args.bootstrap_iters, seed=args.seed)
            base = next((c for c in cells if c["quality"] == q and c["performance"] == p), None)
            boot_rows.append(
                {
                    "quality": q,
                    "performance": p,
                    "rs": base["rs"] if base else None,
                    "ci_low": ci[0] if ci else None,
                    "ci_high": ci[1] if ci else None,
                    "iters": args.bootstrap_iters,
                }
            )
    with (out_dir / "bootstrap_ci.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["quality", "performance", "rs", "ci_low", "ci_high", "iters"])
        w.writeheader()
        for row in boot_rows:
            w.writerow(row)

    # Table 2 (per system, merged)
    agg = []
    for system in sorted({m.get("system") for m in merged}):
        sub = [m for m in merged if m.get("system") == system]
        row = {"system": system, "n_samples": len(sub)}
        for col in ["er", "rr", "density", "ep", "der", "kgc", "rh_at_5", "dcr_at_5", "cig", "cic", "f1", "faithfulness"]:
            vals = [fnum(m.get(col)) for m in sub]
            vals = [v for v in vals if v is not None]
            row[col] = round(float(np.median(vals)), 6) if vals else None
            row[f"{col}_iqr"] = (
                round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 6)
                if len(vals) >= 4
                else None
            )
        agg.append(row)
    with (out_dir / "table2_by_system.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["system", "n_samples"]
        for col in ["er", "rr", "density", "ep", "der", "kgc", "rh_at_5", "dcr_at_5", "cig", "cic", "f1", "faithfulness"]:
            fields += [col, f"{col}_iqr"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in agg:
            w.writerow(row)

    # Figure 1 — ER/KGC vs F1 scatter, colored by system
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for system in sorted({m.get("system") for m in merged}):
        xs = [fnum(m.get("er")) for m in merged if m.get("system") == system]
        ys = [fnum(m.get("f1")) for m in merged if m.get("system") == system]
        pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if pts:
            ax.scatter(*zip(*pts), label=system, alpha=0.8, s=28)
    ax.set_xlabel("Entity Recall / KG-Coverage (ER)")
    ax.set_ylabel("Answer F1")
    ax.set_title("Figure 1 — KG entity coverage vs answer F1 (per sample)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_er_vs_f1.png", dpi=150)
    plt.close(fig)

    # Figure 3 — system × facet heatmap
    facets = ["er", "density", "kgc", "rh_at_5", "cig", "f1"]
    mat = np.zeros((len(agg), len(facets)))
    for i, row in enumerate(agg):
        for j, f in enumerate(facets):
            v = row.get(f)
            mat[i, j] = v if v is not None else 0.0
    fig, ax = plt.subplots(figsize=(7.5, 1.6 * len(agg) + 1.2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(facets)))
    ax.set_xticklabels(["ER", "Density", "KGC", "RH@5", "CiG", "F1"])
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([r["system"] for r in agg])
    for i in range(len(agg)):
        for j in range(len(facets)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.5 else "black")
    ax.set_title("Figure 3 — quality/performance profile by system")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_heatmap.png", dpi=150)
    plt.close(fig)

    summary = {
        "run_dir": str(run_dir),
        "merged_rows": len(merged),
        "significant_pairs": len(significant),
        "outputs": [
            "table3_spearman.csv", "table3_significant.csv", "regression_fixed_effects.csv",
            "bootstrap_ci.csv", "table2_by_system.csv",
            "fig1_er_vs_f1.png", "fig3_heatmap.png",
        ],
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[analysis] merged rows={len(merged)}")
    print(f"[analysis] significant pairs (p<0.05): {len(significant)}")
    for c in sorted(significant, key=lambda x: x["p"] or 1.0)[:8]:
        print(f"  {c['quality']} × {c['performance']}: rs={c['rs']:.3f} p={c['p']:.4f}{sig_stars(c['p'])}")
    if reg:
        for row in reg[:5]:
            print(f"  reg {row['term']}: coef={row['coef']:.3f} p={row['p']:.4f} stdβ={row['std_beta'] if row['std_beta'] is None else round(row['std_beta'], 3)}")
    print(f"[analysis] wrote {len(summary['outputs'])} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
