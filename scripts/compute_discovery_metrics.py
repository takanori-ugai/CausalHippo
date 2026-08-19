#!/usr/bin/env python3
"""SWO69 T3 — discovery metrics (SWO_PLAN.md §4.2, Figure 1/Table 2 inputs).

Joins three sources per (condition, sample):
  1. per_question/<condition>.jsonl — answer metrics + `retrievedContexts`
     (retrieved contexts are only persisted by jars built with the T7
     batch-2 Kotlin changes; rows without the field get RH/DCR/CiC = null)
  2. per-sample KG snapshots        — via the T1 loaders (analyze_kg_quality)
  3. gold annotations + dataset rows

Metrics:
  kgc     — KG-Coverage: share of gold entities present as KG nodes (= T1 ER)
  rh_at_K — Retrieval-Hit@K: share of gold entities appearing in the top-K
            retrieved contexts (K in {1, 3, 5}; null when the run produced
            fewer than K contexts)
  dcr_at_K— Discovery Conversion Rate = RH@K / KGC (KGC > 0)
  cig     — Chain-In-Graph (0/1): every consecutive gold-chain pair has a KG
            path of <= 3 edges AND start–end are connected; causal rows use
            gold-edge presence instead
  cic     — Chain-In-Context (0/1): top-5 context contains every gold-chain
            bridge entity (all chain entities except the final answer);
            cic_all additionally requires the final answer entity
  plr     — Path-Length Ratio: mean KG shortest-path length over the
            gold-chain steps (inf when any step has no path)

Usage:
  python3 scripts/compute_discovery_metrics.py --run-dir eval_results/<ts> \
      [--conditions graphrag,lightrag,...] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_kg_quality import (  # noqa: E402
    DEFAULT_DATA,
    DEFAULT_GOLD,
    build_lookups,
    discover_snapshots,
    find_dataset,
    load_graphrag,
    load_hippo,
    load_json_graph,
    load_youtu,
)
from kg_common import match_entity, surface_in_text  # noqa: E402

CONDITION_TO_SYSTEM = {
    "graphrag": "graphrag",
    "lightrag": "lightrag",
    "pathrag": "pathrag",
    "youturag": "youturag",
    "hipporag_graph": "hipporag",
    "hipporag_dpr": "hipporag",
}

FIELDS = [
    "condition",
    "system",
    "dataset",
    "sample_id",
    "has_gold",
    "n_gold_entities",
    "kgc",
    "rh_at_1",
    "rh_at_3",
    "rh_at_5",
    "dcr_at_1",
    "dcr_at_3",
    "dcr_at_5",
    "cig",
    "cic",
    "cic_all",
    "plr",
    "f1",
    "exact_match",
    "support_recall_at_5",
    "bridge_coverage_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "faithfulness",
    "error",
]

AGG_FIELDS = [
    "condition", "n_samples", "kgc",
    "rh_at_1", "rh_at_3", "rh_at_5",
    "dcr_at_1", "dcr_at_3", "dcr_at_5",
    "cig", "cic", "cic_all", "plr",
    "f1", "exact_match", "support_recall_at_5", "bridge_coverage_at_5",
    "mrr_at_5", "ndcg_at_5", "faithfulness",
]


def load_kg_for(system: str, snap_path: Path):
    if system == "graphrag":
        return load_graphrag(snap_path.parent)
    if system in ("lightrag", "pathrag"):
        payload = json.loads(snap_path.read_text(encoding="utf-8"))
        return load_json_graph(payload, system)
    if system == "hipporag":
        return load_hippo(snap_path.parent)
    if system == "youturag":
        return load_youtu(snap_path.parent)
    raise ValueError(system)


def node_of(entity: str, aliases_map: dict, nodes: list[dict]) -> str | None:
    for n in nodes:
        if match_entity(entity, aliases_map.get(entity), n["surface"]):
            return n["id"]
    return None


def gold_entity_in_text(entity: str, aliases: list[str] | None, text: str) -> bool:
    if surface_in_text(entity, text):
        return True
    return any(surface_in_text(a, text) for a in aliases or [])


def build_graph(nodes: list[dict], edges: list[tuple[str, str]]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(n["id"] for n in nodes)
    for s, t in edges:
        if s != t and s in g and t in g:
            g.add_edge(s, t)
    return g


def compute_chain_metrics(
    graph: nx.Graph,
    nodes: list[dict],
    aliases_map: dict,
    chain: list[str],
    gold_edges: list[dict],
    dataset: str,
) -> tuple[float | None, float | None]:
    """Returns (cig, plr) per §4.2 (0/1 and path-length ratio)."""
    if dataset == "causal" and gold_edges:
        ge = gold_edges[0]
        sn = node_of(ge.get("source") or "", aliases_map, nodes)
        tn = node_of(ge.get("target") or "", aliases_map, nodes)
        if sn is None or tn is None:
            return None, None
        present = sn == tn or graph.has_edge(sn, tn)
        return (1.0 if present else 0.0), (1.0 if present else float("inf"))

    pairs = list(zip(chain, chain[1:]))
    if not pairs:
        return None, None
    start = node_of(chain[0], aliases_map, nodes)
    end = node_of(chain[-1], aliases_map, nodes)
    distances = []
    for a, b in pairs:
        an, bn = node_of(a, aliases_map, nodes), node_of(b, aliases_map, nodes)
        if an is None or bn is None:
            return 0.0, float("inf")
        try:
            distances.append(nx.shortest_path_length(graph, an, bn))
        except nx.NetworkXNoPath:
            return 0.0, float("inf")
    connected = False
    if start is not None and end is not None:
        try:
            nx.shortest_path_length(graph, start, end)
            connected = True
        except nx.NetworkXNoPath:
            connected = False
    cig = 1.0 if (connected and all(d <= 3 for d in distances)) else 0.0
    return cig, sum(distances) / len(distances)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--conditions", default="", help="comma list (default: all present)")
    ap.add_argument("--out", type=Path, default=None, help="default <run-dir>/discovery_metrics")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run_dir
    pq_dir = run_dir / "per_question"
    if not pq_dir.is_dir():
        print(f"error: {pq_dir} not found", file=sys.stderr)
        return 2

    lookups = build_lookups(DEFAULT_DATA, DEFAULT_GOLD)
    found = discover_snapshots(run_dir)

    rows: list[dict] = []
    skipped: list[str] = []
    for cf in sorted(pq_dir.glob("*.jsonl")):
        condition = cf.stem
        system = CONDITION_TO_SYSTEM.get(condition)
        if system is None:
            continue
        if args.conditions and condition not in args.conditions.split(","):
            continue
        for line in cf.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row_in = json.loads(line)
            sample_id = str(row_in.get("sampleId") or row_in.get("sample_id") or "")
            out: dict = {
                "condition": condition,
                "system": system,
                "dataset": "",
                "sample_id": sample_id,
                "has_gold": False,
                "n_gold_entities": None,
                "kgc": None,
                "rh_at_1": None, "rh_at_3": None, "rh_at_5": None,
                "dcr_at_1": None, "dcr_at_3": None, "dcr_at_5": None,
                "cig": None, "cic": None, "cic_all": None, "plr": None,
                "f1": row_in.get("f1"),
                "exact_match": row_in.get("exactMatch"),
                "support_recall_at_5": row_in.get("supportRecallAt5"),
                "bridge_coverage_at_5": row_in.get("bridgeCoverageAt5"),
                "mrr_at_5": row_in.get("mrrAt5"),
                "ndcg_at_5": row_in.get("ndcgAt5"),
                "faithfulness": row_in.get("faithfulness"),
                "error": row_in.get("error"),
            }
            rows.append(out)

            hit = find_dataset(sample_id, lookups)
            if hit is None:
                skipped.append(f"{condition}/{sample_id}: no dataset row")
                continue
            dataset, (_data_row, gold, _corpus) = hit
            out["dataset"] = dataset
            if not (gold and gold.get("has_gold") and gold.get("gold_entities")):
                continue
            gold_entities = gold["gold_entities"]
            aliases_map = gold.get("gold_entity_aliases") or {}
            chain = gold.get("gold_chain") or []
            gold_edges = gold.get("gold_edges") or []
            out["has_gold"] = True
            out["n_gold_entities"] = len(gold_entities)

            # --- KG side: KGC, CiG, PLR -------------------------------------
            snap = found.get(system, {}).get(sample_id)
            nodes: list[dict] = []
            if snap is not None:
                try:
                    nodes, edges, _extras = load_kg_for(system, snap)
                except Exception as exc:  # noqa: BLE001
                    skipped.append(f"{condition}/{sample_id}: KG load failed: {exc}")
                    continue
                matched = sum(1 for e in gold_entities if node_of(e, aliases_map, nodes) is not None)
                out["kgc"] = matched / len(gold_entities)
                graph = build_graph(nodes, edges)
                cig, plr = compute_chain_metrics(
                    graph, nodes, aliases_map, chain, gold_edges, dataset
                )
                out["cig"] = cig
                out["plr"] = plr

            # --- retrieval side: RH@K, DCR, CiC (needs retrievedContexts) ----
            contexts = row_in.get("retrievedContexts") or []
            if contexts:
                for k in (1, 3, 5):
                    if len(contexts) < k:
                        continue
                    topk_text = "\n".join(contexts[:k])
                    hits = sum(
                        1
                        for e in gold_entities
                        if gold_entity_in_text(e, aliases_map.get(e), topk_text)
                    )
                    out[f"rh_at_{k}"] = hits / len(gold_entities)
                    if out["kgc"]:
                        out[f"dcr_at_{k}"] = out[f"rh_at_{k}"] / out["kgc"]

                ctx_text = "\n".join(contexts)
                if len(chain) >= 2:
                    bridges = chain[:-1]
                    out["cic"] = float(
                        all(gold_entity_in_text(b, aliases_map.get(b), ctx_text) for b in bridges)
                    )
                    out["cic_all"] = float(
                        all(gold_entity_in_text(e, aliases_map.get(e), ctx_text) for e in chain)
                    )
                elif gold_edges:
                    ge = gold_edges[0]
                    ok = all(
                        gold_entity_in_text(
                            ge.get(key, ""), aliases_map.get(ge.get(key, "")), ctx_text
                        )
                        for key in ("source", "target")
                    )
                    out["cic"] = float(ok)
                    out["cic_all"] = float(ok)

    out_dir = args.out or (run_dir / "discovery_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "discovery_metrics_per_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    agg_rows = []
    for cond in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cond]
        agg: dict = {"condition": cond, "n_samples": len(sub)}
        for col in AGG_FIELDS:
            if col in ("condition", "n_samples"):
                continue
            vals = [
                r[col]
                for r in sub
                if isinstance(r.get(col), (int, float))
                and math.isfinite(r[col])  # skip NaN and inf (plr=inf = disconnected chain)
            ]
            agg[col] = round(sum(vals) / len(vals), 6) if vals else None
        agg_rows.append(agg)
    agg_path = out_dir / "discovery_metrics_by_condition.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=AGG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in agg_rows:
            writer.writerow(row)

    (out_dir / "discovery_metrics_summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "rows_written": len(rows),
                "skipped": skipped,
                "outputs": [str(csv_path), str(agg_path)],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[discovery] rows={len(rows)} "
        f"conditions={sorted({r['condition'] for r in rows})} skipped={len(skipped)}"
    )
    for s in skipped[:10]:
        print(f"  skipped {s}")
    print(f"[discovery] wrote {csv_path}")
    print(f"[discovery] wrote {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
