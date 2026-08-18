#!/usr/bin/env python3
"""SWO69 T1 — per-system KG quality analysis (SWO_PLAN.md §3.1–§3.3, Table 1/Table 2 inputs).

Loads every per-sample KG snapshot found under a run directory, normalizes all
five systems to (nodes, edges) and computes:

  §3.3 structural metrics (gold-free, all datasets):
      n_nodes, n_edges, density, avg_degree, max_degree, degree_gini,
      n_components, r_lcc, inr, smr
  §3.2 extraction-quality metrics (where gold annotations exist):
      er  — entity recall over gold entities (+ aliases), §3.2
      ep  — entity precision as document-grounding rate (surface form of each
            extracted node appears in the sample passages; the LLM-judge arm
            of §3.2 EP is out of scope for the offline box)
      der — duplicate entity ratio (near-duplicate node pair count / |V|)
      rr  — relation recall (causal: gold edge present; MuSiQue: each
            consecutive gold-chain pair connected by a path of ≤ 3 edges);
            causal rows additionally carry rr_true / rr_false splits
      ser — synonym (KNN) edge ratio; HippoRAG-only. Heuristic: KNN synonym
            edges are weighted by embedding similarity (< 1.0) while OpenIE
            edge weights are co-occurrence counts (≥ 1), see
            hipporag/HippoRag.kt addSynonymyEdges/addNewEdges.

Snapshot layouts (produced by the T7 Kotlin snapshot hooks):
  GraphRAG   <sample_workdir>/kg_snapshot/{entities,relationships}.parquet
  LightRAG   <sample_workdir>/kg_snapshot/knowledge-graph.json   ({nodes:[…], edges:[…]})
  PathRAG    <sample_workdir>/kg_snapshot/knowledge-graph.json   ({nodes:{id:…}, edges:[…]})
  HippoRAG   workdirs/<suffix>/snapshots/<sample>/working_dir/graph.json
  YoutuRAG   <sample_workdir>/kg_snapshot/<dataset>_new.json     ([{start_node, relation, end_node}])

Outputs (default <run-dir>/kg_quality/):
  kg_quality_per_sample.csv, kg_quality_by_system.csv, kg_quality_summary.json

Usage:
  python3 scripts/analyze_kg_quality.py --run-dir eval_results/<ts> [--systems a,b,c]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_common import match_entity, normalize_surface, surface_in_text, word_boundary_substring  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

DEFAULT_DATA = {
    "musique": REPO / "data/musique_experiment/musique_dev_balanced_30.jsonl",
    "causal": REPO / "data/causal_experiment/causal_qa_balanced_30.jsonl",
    "webis": REPO / "data/webis_experiment/webis_train_balanced_30.jsonl",
}
DEFAULT_GOLD = {
    "musique": REPO / "data/gold_annotations/musique_gold_30.jsonl",
    "causal": REPO / "data/gold_annotations/causal_gold_30.jsonl",
    "webis": REPO / "data/gold_annotations/webis_gold_30.jsonl",
}

STRUCT_COLS = [
    "n_nodes", "n_edges", "density", "avg_degree", "max_degree", "degree_gini",
    "n_components", "r_lcc", "inr", "smr",
]
GOLD_COLS = ["ser", "er", "ep", "der", "rr", "rr_true", "rr_false"]
META_COLS = ["n_gold_entities", "n_gold_edges", "corpus_chars"]

ROW_FIELDS = [
    "system", "dataset", "sample_id", "snapshot_path",
    *STRUCT_COLS, *GOLD_COLS, *META_COLS,
]


def norm_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


# --------------------------------------------------------------------------- #
# Loaders: system snapshot -> (nodes, edges, extras)
# nodes: [{"id": str, "surface": str}]  edges: [(src_id, dst_id)]
# --------------------------------------------------------------------------- #


def load_graphrag(snapshot_dir: Path):
    import pyarrow.parquet as pq

    ents = pq.read_table(snapshot_dir / "entities.parquet").to_pydict()
    id_to_surface = {
        str(row_id): str(name or row_id)
        for row_id, name in zip(ents.get("id", []), ents.get("name", []))
    }
    nodes = [{"id": nid, "surface": surf} for nid, surf in id_to_surface.items()]
    edges = []
    rels = pq.read_table(snapshot_dir / "relationships.parquet").to_pydict()
    for src, dst in zip(rels.get("sourceId", []), rels.get("targetId", [])):
        if src in id_to_surface and dst in id_to_surface:
            edges.append((src, dst))
    return nodes, edges, {}


def load_json_graph(payload: dict, system: str):
    """LightRAG ({nodes: [{id, …}], edges: [{source, target}]}),
    PathRAG ({nodes: {id: {…}}, edges: [{source, target, data}]})."""
    raw_nodes = payload.get("nodes")
    nodes = []
    if isinstance(raw_nodes, dict):  # PathRAG
        for nid, data in raw_nodes.items():
            data = data if isinstance(data, dict) else {}
            surface = str(data.get("entity_name") or data.get("name") or nid)
            nodes.append({"id": str(nid), "surface": surface})
    elif isinstance(raw_nodes, list):  # LightRAG
        for item in raw_nodes:
            if not isinstance(item, dict):
                continue
            nid = item.get("id") or item.get("entity_name") or item.get("name")
            if nid is None:
                continue
            nid = str(nid)
            surface = str(item.get("entity_name") or item.get("name") or nid)
            nodes.append({"id": nid, "surface": surface})
    else:
        raw_nodes = []
    seen = set()
    dedup = []
    for n in nodes:
        if n["id"] not in seen:
            seen.add(n["id"])
            dedup.append(n)
    nodes = dedup
    ids = {n["id"] for n in nodes}
    edges = []
    for e in payload.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src = e.get("source") or e.get("src") or e.get("start_node")
        dst = e.get("target") or e.get("tgt") or e.get("end_node")
        if src is None or dst is None:
            continue
        src, dst = str(src), str(dst)
        if src in ids and dst in ids:
            edges.append((src, dst))
    return nodes, edges, {"source": system}


def load_hippo(snapshot_dir: Path):
    graph = json.loads((snapshot_dir / "graph.json").read_text(encoding="utf-8"))
    vertices = graph.get("vertices") or []
    nodes = [
        {
            "id": str(v.get("hash_id") or f"vertex_{i}"),
            "surface": str(v.get("content") or v.get("name") or v.get("hash_id") or f"vertex_{i}"),
        }
        for i, v in enumerate(vertices)
    ]
    by_idx = {i: n["id"] for i, n in enumerate(nodes)}
    edges = []
    weights = []
    for e in graph.get("edges") or []:
        try:
            src = by_idx[int(e["source"])]
            dst = by_idx[int(e["target"])]
        except (KeyError, TypeError, ValueError):
            continue
        edges.append((src, dst))
        weights.append(float(e.get("weight", 1.0)))
    return nodes, edges, {"edge_weights": weights}


def load_youtu(snapshot_dir: Path):
    files = sorted(snapshot_dir.glob("*_new.json"))
    if not files:
        raise FileNotFoundError(f"no *_new.json graph under {snapshot_dir}")
    triples = json.loads(files[0].read_text(encoding="utf-8"))
    nodes: dict[str, dict] = {}
    edges = []
    for t in triples:
        if not isinstance(t, dict):
            continue
        start = t.get("start_node") or {}
        end = t.get("end_node") or {}
        sl, el = start.get("label"), end.get("label")
        if not sl or not el:
            continue
        nodes.setdefault(sl, {"id": sl, "surface": sl})
        nodes.setdefault(el, {"id": el, "surface": el})
        edges.append((sl, el))
    return list(nodes.values()), edges, {}


# --------------------------------------------------------------------------- #
# Snapshot discovery under a run directory
# --------------------------------------------------------------------------- #

SNAPSHOT_PATTERNS = {
    # (glob, kind): kind determines loader and where the sample id sits in parts
    "graphrag": ("**/kg_snapshot/entities.parquet", "file"),
    "lightrag": ("**/kg_snapshot/knowledge-graph.json", "file"),
    "pathrag": ("**/kg_snapshot/knowledge-graph.json", "file"),
    "youturag": ("**/kg_snapshot/*_new.json", "file"),
    "hipporag": ("**/working_dir/graph.json", "file"),
}


def discover_snapshots(run_dir: Path) -> dict[str, dict[str, Path]]:
    """system -> {sample_id: snapshot_path}. Keeps the newest snapshot per id.

    LightRAG and PathRAG share the knowledge-graph.json layout, so a snapshot
    only counts for a system when its workdir suffix belongs to that system.
    """
    found: dict[str, dict[str, Path]] = {s: {} for s in SNAPSHOT_PATTERNS}
    for system, (pattern, _kind) in SNAPSHOT_PATTERNS.items():
        for path in run_dir.glob(pattern):
            if not path.is_file():
                continue
            sample_id = sample_id_from_path(path, system)
            if sample_id is None:
                continue
            suffix = workdir_suffix_from_path(path, system, sample_id)
            if suffix is not None and not (
                suffix == system
                or suffix.startswith(f"{system}_")
                or f"_{system}" in suffix
            ):
                continue
            prev = found[system].get(sample_id)
            if prev is None or path.stat().st_mtime >= prev.stat().st_mtime:
                found[system][sample_id] = path
    return found


def workdir_suffix_from_path(path: Path, system: str, sample_id: str) -> str | None:
    parts = list(path.parts)
    if system == "hipporag":
        if "snapshots" in parts:
            i = parts.index("snapshots")
            return parts[i - 1] if i >= 1 else None
        # Legacy last-sample-wins layout: …/<suffix>/<model>/graph.json
        return parts[-3] if len(parts) >= 3 else None
    if sample_id in parts:
        i = parts.index(sample_id)
        return parts[i - 1] if i >= 1 else None
    return None


_PERTURBATION_SEG = re.compile(r"^p\d+(_\d+)?$")


def _is_pert_seg(seg: str) -> bool:
    return bool(_PERTURBATION_SEG.fullmatch(seg))


def sample_id_from_path(path: Path, system: str) -> str | None:
    """Recover the sample id from a snapshot path.

    Layouts (pert = optional E2 perturbation segment, e.g. p1_30):
      …/<suffix>/<sampleId>[/<pert>]/kg_snapshot/<file>          (file formats)
      …/<suffix>/snapshots/<sampleId>/working_dir/graph.json      (hipporag T7)
      …/<suffix>/<sampleId>[/<pert>]/working_dir/graph.json       (hipporag E2)
      …/<suffix>/<modelSuffix>/graph.json                         (hipporag legacy)
    """
    parts = list(path.parts)
    if system in ("graphrag", "lightrag", "pathrag", "youturag"):
        if "kg_snapshot" in parts:
            i = parts.index("kg_snapshot")
            j = i - 1
            if _is_pert_seg(parts[j]):
                j -= 1
            return parts[j] if j >= 2 else None
    elif system == "hipporag":
        if "working_dir" in parts:
            i = parts.index("working_dir")
            j = i - 1
            if _is_pert_seg(parts[j]):
                return parts[j - 1] if j - 1 >= 2 else None
            return parts[j] if j >= 1 else None
        if len(parts) >= 2:
            cand = parts[-2]
            if ":" not in cand and cand not in ("workdirs", "output"):
                return cand
    return None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def gini_coefficient(values: list[float]) -> float:
    v = sorted(values)
    n = len(v)
    total = sum(v)
    if n == 0 or total <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2.0 * cum) / (n * total) - (n + 1.0) / n


def structural_metrics(nodes: list[dict], edges: list[tuple[str, str]]) -> dict[str, float]:
    n_nodes = len(nodes)
    n_edges = len(edges)
    g = nx.Graph()
    g.add_nodes_from(n["id"] for n in nodes)
    self_loops = 0
    multi = 0
    pair_seen: set[frozenset] = set()
    for src, dst in edges:
        if src == dst:
            self_loops += 1
            continue
        key = frozenset((src, dst))
        if key in pair_seen:
            multi += 1
        pair_seen.add(key)
        if src in g and dst in g:
            g.add_edge(src, dst)
    n = g.number_of_nodes()
    degrees = list(dict(g.degree()).values())
    density = (2.0 * len(g.edges)) / (n * (n - 1)) if n > 1 else 0.0
    isolated = sum(1 for d in degrees if d == 0)
    components = sorted((len(c) for c in nx.connected_components(g)), reverse=True) if n else []
    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "density": density,
        "avg_degree": (2.0 * len(g.edges)) / n if n else 0.0,
        "max_degree": float(max(degrees)) if degrees else 0.0,
        "degree_gini": gini_coefficient(degrees),
        "n_components": float(len(components)),
        "r_lcc": (components[0] / n) if n and components else 0.0,
        "inr": isolated / n if n else 0.0,
        "smr": (self_loops + multi) / n_edges if n_edges else 0.0,
    }


def match_gold_to_node(
    gold_entity: str,
    aliases: list[str] | None,
    nodes: list[dict],
) -> str | None:
    for node in nodes:
        if match_entity(gold_entity, aliases, node["surface"]):
            return node["id"]
    return None


def path_exists_within(g: nx.Graph, a: str, b: str, max_len: int) -> bool:
    if a not in g or b not in g:
        return False
    if a == b:
        return True
    try:
        return nx.shortest_path_length(g, a, b) <= max_len
    except nx.NetworkXNoPath:
        return False


def duplicate_ratio(nodes: list[dict]) -> float:
    n = len(nodes)
    if n < 2:
        return 0.0
    pairs = 0
    norm = [(nid, normalize_surface(node["surface"])) for nid, node in ((x["id"], x) for x in nodes)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = norm[i][1], norm[j][1]
            if not a or not b or a == b:
                pairs += 1
                continue
            longer, shorter = (a, b) if len(a) >= len(b) else (b, a)
            if len(shorter) >= 4 and word_boundary_substring(longer, shorter):
                pairs += 1
    return pairs / n


def gold_metrics(
    system: str,
    dataset: str,
    nodes: list[dict],
    edges: list[tuple[str, str]],
    extras: dict,
    gold: dict | None,
    corpus_text: str,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {c: None for c in GOLD_COLS}
    out["n_gold_entities"] = 0.0
    out["n_gold_edges"] = 0.0
    out["corpus_chars"] = float(len(corpus_text))

    if system == "hipporag" and edges:
        weights = extras.get("edge_weights") or []
        out["ser"] = (
            sum(1.0 for w in weights if w < 1.0) / len(edges) if weights else None
        )

    if not gold or not gold.get("has_gold"):
        return out

    aliases_map = gold.get("gold_entity_aliases") or {}
    gold_entities = gold.get("gold_entities") or []
    g = nx.Graph()
    g.add_nodes_from(n["id"] for n in nodes)
    pair_seen = set()
    for src, dst in edges:
        if src != dst and src in g and dst in g:
            g.add_edge(src, dst)
            pair_seen.add(frozenset((src, dst)))

    # ER
    if gold_entities:
        matched = sum(
            1 for e in gold_entities if match_gold_to_node(e, aliases_map.get(e), nodes) is not None
        )
        out["er"] = matched / len(gold_entities)
    out["n_gold_entities"] = float(len(gold_entities))

    # EP (document-grounding rate)
    if nodes and corpus_text:
        grounded = sum(1 for n in nodes if surface_in_text(n["surface"], corpus_text))
        out["ep"] = grounded / len(nodes)

    # DER
    out["der"] = duplicate_ratio(nodes)

    # RR
    if dataset == "causal":
        gold_edges = gold.get("gold_edges") or []
        out["n_gold_edges"] = float(len(gold_edges))
        if gold_edges:
            hits = 0
            true_total = true_hits = false_total = false_hits = 0
            for ge in gold_edges:
                s = ge.get("source") or ""
                t = ge.get("target") or ""
                sn = match_gold_to_node(s, aliases_map.get(s), nodes)
                tn = match_gold_to_node(t, aliases_map.get(t), nodes)
                present = bool(sn and tn and (sn == tn or frozenset((sn, tn)) in pair_seen))
                label_true = bool(ge.get("label_true", gold.get("label_true", True)))
                if label_true:
                    true_total += 1
                    true_hits += int(present)
                else:
                    false_total += 1
                    false_hits += int(present)
                hits += int(present)
            out["rr"] = hits / len(gold_edges)
            out["rr_true"] = true_hits / true_total if true_total else None
            out["rr_false"] = false_hits / false_total if false_total else None
    else:
        chain = gold.get("gold_chain") or []
        pairs = list(zip(chain, chain[1:]))
        out["n_gold_edges"] = float(len(pairs))
        if pairs:
            hits = 0
            for a, b in pairs:
                an = match_gold_to_node(a, aliases_map.get(a), nodes)
                bn = match_gold_to_node(b, aliases_map.get(b), nodes)
                hits += int(an is not None and bn is not None and path_exists_within(g, an, bn, 3))
            out["rr"] = hits / len(pairs)

    return out


# --------------------------------------------------------------------------- #
# Datasets / gold lookup
# --------------------------------------------------------------------------- #


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_lookups(data_paths: dict, gold_paths: dict) -> dict:
    """dataset -> {norm_id: (data_row, gold_row|None, corpus_text)}"""
    lookups = {}
    for ds, dp in data_paths.items():
        gold_by_id = {}
        gp = gold_paths.get(ds)
        if gp and gp.exists():
            for g in load_jsonl(gp):
                gold_by_id[norm_id(g.get("id"))] = g
        table = {}
        if dp and dp.exists():
            for row in load_jsonl(dp):
                corpus = " ".join(
                    str(p.get("paragraph_text", "")) for p in row.get("paragraphs") or []
                )
                table[norm_id(row.get("id"))] = (row, gold_by_id.get(norm_id(row.get("id"))), corpus)
        lookups[ds] = table
    return lookups


def find_dataset(key: str, lookups: dict) -> tuple[str, dict] | None:
    nk = norm_id(key)
    for ds, table in lookups.items():
        if nk in table:
            return ds, table[nk]
    # Fuzzy fallback: normalized containment (sanitized ids vs source ids)
    for ds, table in lookups.items():
        for tid, entry in table.items():
            if (nk and tid and (nk in tid or tid in nk)) and min(len(nk), len(tid)) >= 8:
                return ds, entry
    return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--systems", default="", help="comma list (default: auto-detect from snapshots)")
    ap.add_argument("--out", type=Path, default=None, help="default <run-dir>/kg_quality")
    for ds in ("musique", "causal", "webis"):
        ap.add_argument(f"--data-{ds}", type=Path, default=DEFAULT_DATA[ds])
        ap.add_argument(f"--gold-{ds}", type=Path, default=DEFAULT_GOLD[ds])
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"error: run dir not found: {run_dir}", file=sys.stderr)
        return 2

    data_paths = {
        "musique": args.data_musique,
        "causal": args.data_causal,
        "webis": args.data_webis,
    }
    gold_paths = {
        "musique": args.gold_musique,
        "causal": args.gold_causal,
        "webis": args.gold_webis,
    }
    lookups = build_lookups(data_paths, gold_paths)

    found = discover_snapshots(run_dir)
    wanted = [s.strip() for s in args.systems.split(",") if s.strip()] or list(SNAPSHOT_PATTERNS)
    unknown = [s for s in wanted if s not in SNAPSHOT_PATTERNS]
    if unknown:
        print(f"error: unknown system(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    rows = []
    skipped = []
    for system in wanted:
        for sample_id, snap_path in sorted(found[system].items()):
            try:
                if system == "graphrag":
                    nodes, edges, extras = load_graphrag(snap_path.parent)
                elif system in ("lightrag", "pathrag"):
                    payload = json.loads(snap_path.read_text(encoding="utf-8"))
                    nodes, edges, extras = load_json_graph(payload, system)
                elif system == "hipporag":
                    nodes, edges, extras = load_hippo(snap_path.parent)
                elif system == "youturag":
                    nodes, edges, extras = load_youtu(snap_path.parent)
                else:
                    raise ValueError(system)
            except Exception as exc:  # noqa: BLE001
                skipped.append((system, sample_id, f"load failed: {exc}"))
                continue

            hit = find_dataset(sample_id, lookups)
            if hit is None:
                skipped.append((system, sample_id, "no matching dataset row"))
                continue
            dataset, (_data_row, gold, corpus_text) = hit

            row = {
                "system": system,
                "dataset": dataset,
                "sample_id": sample_id,
                "snapshot_path": str(snap_path),
            }
            row.update(structural_metrics(nodes, edges))
            row.update(gold_metrics(system, dataset, nodes, edges, extras, gold, corpus_text))
            rows.append(row)

    out_dir = args.out or (run_dir / "kg_quality")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_csv = out_dir / "kg_quality_per_sample.csv"
    with per_sample_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    by_system_rows = []
    systems = sorted({r["system"] for r in rows})
    for system in systems:
        sub = [r for r in rows if r["system"] == system]
        agg = {"system": system, "n_samples": len(sub)}
        for col in STRUCT_COLS + GOLD_COLS + META_COLS:
            vals = [r[col] for r in sub if isinstance(r.get(col), (int, float))]
            agg[col] = round(sum(vals) / len(vals), 6) if vals else None
        by_system_rows.append(agg)
    by_system_csv = out_dir / "kg_quality_by_system.csv"
    with by_system_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["system", "n_samples", *STRUCT_COLS, *GOLD_COLS, *META_COLS],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in by_system_rows:
            writer.writerow(row)

    summary = {
        "run_dir": str(run_dir),
        "systems": {
            s: {"n_snapshots": len(found.get(s, {}))} for s in SNAPSHOT_PATTERNS
        },
        "rows_written": len(rows),
        "skipped": skipped,
        "outputs": [str(per_sample_csv), str(by_system_csv)],
    }
    (out_dir / "kg_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[analyze_kg_quality] rows={len(rows)} systems={systems} skipped={len(skipped)}")
    for s, sid, why in skipped[:20]:
        print(f"  skipped {s}/{sid}: {why}")
    print(f"[analyze_kg_quality] wrote {per_sample_csv}")
    print(f"[analyze_kg_quality] wrote {by_system_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
