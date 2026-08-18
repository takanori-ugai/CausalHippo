#!/usr/bin/env python3
"""SWO69 T4 — KG post-hoc perturbation tool (SWO_PLAN.md §5 E2, P0–P5).

Reads a per-sample KG snapshot (any of the five system formats), applies one
perturbation, and writes the perturbed graph back in the SAME format/layout so
the E2 Kotlin runner (T7 `--no-rebuild` graph-load mode) can consume it and the
T1/T3 analysis scripts can audit it:

  P0    control (identity)
  P1    random edge removal            p1_10 / p1_30 / p1_50 (10/30/50 %)
  P2    random node removal (+edges)   p2_10 / p2_30   (10/30 %)
  P3    selective gold-entity removal  p3  (gold nodes only, needs --gold-file/--sample-id)
  P4    description/text removal       p4  (structure kept; no-op where the format has none)
  P5    fake node+edge injection       p5_10 / p5_30   (10/30 % of node count)

All stochastic operations are seeded (--seed, default 42) for reproducibility.

Usage:
  python3 scripts/perturb_kg.py \
      --system pathrag --snapshot eval_results/<run>/workdirs/pathrag/<sid>/kg_snapshot/knowledge-graph.json \
      --perturbation p1_30 --out eval_results/swo69_e2/pathrag/<sid>/p1_30/kg_snapshot/knowledge-graph.json \
      [--seed 42]

Directory-based snapshots (graphrag entities.parquet, hipporag working_dir/graph.json,
youturag *_new.json) take the file path as --snapshot; --out must be the
corresponding file in the output location.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kg_common import match_entity  # noqa: E402

PERTURBATIONS = {
    "p0": ("none", None),
    "p1_10": ("edges", 0.10),
    "p1_30": ("edges", 0.30),
    "p1_50": ("edges", 0.50),
    "p2_10": ("nodes", 0.10),
    "p2_30": ("nodes", 0.30),
    "p3": ("gold_nodes", None),
    "p4": ("descriptions", None),
    "p5_10": ("noise", 0.10),
    "p5_30": ("noise", 0.30),
}


class KG:
    """System-agnostic graph: nodes {id -> {surface, description, raw}},
    edges [{source, target, description, raw}]."""

    def __init__(self, system: str):
        self.system = system
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.input_path: Path | None = None
        self.perturbation: str = "p0"

    def remove_nodes(self, ids: set[str]) -> None:
        for i in ids:
            self.nodes.pop(i, None)
        self.edges = [
            e for e in self.edges if e["source"] not in ids and e["target"] not in ids
        ]

    def remove_edge_indexes(self, idxs: set[int]) -> None:
        self.edges = [e for i, e in enumerate(self.edges) if i not in idxs]


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #


def _node(nid: str, surface: str, description: str, raw: dict) -> dict:
    return {"id": nid, "surface": surface, "description": description, "raw": raw}


def read_kg(system: str, snapshot_path: Path) -> KG:
    kg = KG(system)
    kg.input_path = snapshot_path
    if system == "graphrag":
        import pyarrow.parquet as pq

        d = snapshot_path.parent
        ents = pq.read_table(d / "entities.parquet").to_pydict()
        summaries = {}
        try:
            s = pq.read_table(d / "entity_summaries.parquet").to_pydict()
            summaries = dict(zip(s.get("entityId", []), s.get("summary", [])))
        except FileNotFoundError:
            pass
        for eid, name in zip(ents.get("id", []), ents.get("name", [])):
            kg.nodes[str(eid)] = _node(str(eid), str(name or eid), summaries.get(eid, ""), {})
        rels = pq.read_table(d / "relationships.parquet").to_pydict()
        for src, dst, desc in zip(rels.get("sourceId", []), rels.get("targetId", []), rels.get("description", [])):
            kg.edges.append({"source": str(src), "target": str(dst), "description": desc or "", "raw": {}})
        return kg
    if system in ("lightrag", "pathrag"):
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
        raw_nodes = payload.get("nodes")
        if isinstance(raw_nodes, dict):  # PathRAG
            for nid, data in raw_nodes.items():
                data = data if isinstance(data, dict) else {}
                kg.nodes[str(nid)] = _node(
                    str(nid),
                    str(data.get("entity_name") or data.get("name") or nid),
                    str(data.get("description") or ""),
                    data,
                )
        else:  # LightRAG
            for item in raw_nodes or []:
                if not isinstance(item, dict):
                    continue
                nid = str(item.get("id") or item.get("entity_name") or item.get("name") or "")
                if not nid:
                    continue
                kg.nodes[nid] = _node(
                    nid,
                    str(item.get("entity_name") or item.get("name") or nid),
                    str(item.get("description") or ""),
                    item,
                )
        for e in payload.get("edges") or []:
            if not isinstance(e, dict):
                continue
            src = e.get("source") or e.get("src")
            dst = e.get("target") or e.get("tgt")
            if src is None or dst is None:
                continue
            desc = e.get("description") or (e.get("data") or {}).get("description") or ""
            kg.edges.append({"source": str(src), "target": str(dst), "description": desc, "raw": e})
        return kg
    if system == "hipporag":
        graph = json.loads(snapshot_path.read_text(encoding="utf-8"))
        vertices = graph.get("vertices") or []
        for i, v in enumerate(vertices):
            nid = str(v.get("hash_id") or f"vertex_{i}")
            kg.nodes[nid] = _node(
                nid,
                str(v.get("content") or v.get("name") or nid),
                "",
                v,
            )
        by_idx = {i: n["id"] for i, n in enumerate(kg.nodes.values())}
        for e in graph.get("edges") or []:
            try:
                src = by_idx[int(e["source"])]
                dst = by_idx[int(e["target"])]
            except (KeyError, TypeError, ValueError):
                continue
            kg.edges.append({"source": src, "target": dst, "description": "", "raw": e})
        return kg
    if system == "youturag":
        triples = json.loads(snapshot_path.read_text(encoding="utf-8"))
        for t in triples:
            if not isinstance(t, dict):
                continue
            sl = (t.get("start_node") or {}).get("label")
            el = (t.get("end_node") or {}).get("label")
            if not sl or not el:
                continue
            for label in (sl, el):
                if label not in kg.nodes:
                    kg.nodes[label] = _node(label, label, "", {})
            kg.edges.append(
                {
                    "source": sl,
                    "target": el,
                    "description": str(t.get("relation") or ""),
                    "raw": t,
                }
            )
        return kg
    raise ValueError(system)


# --------------------------------------------------------------------------- #
# Writers (same format/layout as the original)
# --------------------------------------------------------------------------- #


def write_kg(kg: KG, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if kg.system == "graphrag":
        import pyarrow as pa
        import pyarrow.parquet as pq

        in_dir = Path(kg.input_path).parent
        out_dir = snapshot_path.parent
        id_idx = None
        # Entities
        ent_table = pq.read_table(in_dir / "entities.parquet")
        rows = list(zip(*[ent_table[c].to_pylist() for c in ent_table.column_names]))
        id_idx = ent_table.column_names.index("id")
        kept = [r for r in rows if str(r[id_idx]) in kg.nodes]
        col_lists = {c: [] for c in ent_table.column_names}
        for r in kept:
            for c, v in zip(ent_table.column_names, r):
                col_lists[c].append(v)
        if "name" in col_lists:
            col_lists["name"] = [kg.nodes[str(r[id_idx])]["surface"] for r in kept]
        pq.write_table(
            pa.table({c: pa.array(col_lists[c], type=ent_table.column(c).type) for c in ent_table.column_names}),
            out_dir / "entities.parquet",
        )
        # Relationships
        rel_table = pq.read_table(in_dir / "relationships.parquet")
        rel_names = rel_table.column_names
        rs = rel_names.index("sourceId")
        rt = rel_names.index("targetId")
        rdesc = rel_names.index("description") if "description" in rel_names else None
        rel_rows = list(zip(*[rel_table[c].to_pylist() for c in rel_names]))
        kept_rows = [r for r in rel_rows if str(r[rs]) in kg.nodes and str(r[rt]) in kg.nodes]
        col_lists = {c: [] for c in rel_names}
        for r in kept_rows:
            for c, v in zip(rel_names, r):
                col_lists[c].append(v)
        if kg.perturbation == "p4" and rdesc is not None:
            col_lists["description"] = ["" for _ in kept_rows]
        pq.write_table(
            pa.table({c: pa.array(col_lists[c], type=rel_table.column(c).type) for c in rel_names}),
            out_dir / "relationships.parquet",
        )
        # P4 support: clear entity summaries if the file exists.
        if kg.perturbation == "p4":
            sum_in = in_dir / "entity_summaries.parquet"
            if sum_in.exists():
                st = pq.read_table(sum_in)
                cols = {c: st[c].to_pylist() for c in st.column_names}
                if "summary" in cols:
                    cols["summary"] = ["" for _ in cols["summary"]]
                pq.write_table(
                    pa.table({c: pa.array(cols[c], type=st.column(c).type) for c in st.column_names}),
                    out_dir / "entity_summaries.parquet",
                )
        # Carry over the non-structural artifacts the query path may need.
        for extra in ("text_units.parquet", "text_embeddings.parquet", "entity_embeddings.parquet",
                      "communities.parquet", "community_reports.json", "vector_store.json", "claims.parquet"):
            src = in_dir / extra
            dst = out_dir / extra
            if src.exists() and not dst.exists():
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        return

    if kg.system in ("lightrag", "pathrag"):
        if kg.system == "pathrag":
            nodes_out = {
                nid: _with_description(node["raw"], node["description"])
                for nid, node in kg.nodes.items()
            }
        else:
            nodes_out = [
                {**node["raw"], "id": nid, "description": node["description"]}
                for nid, node in kg.nodes.items()
            ]
        payload = {
            "nodes": nodes_out,
            "edges": [
                (
                    {"source": e["source"], "target": e["target"], "description": e["description"]}
                    if kg.system == "lightrag"
                    else {**e["raw"], "source": e["source"], "target": e["target"]}
                )
                for e in kg.edges
            ],
            "metadata": {"nodeCount": len(kg.nodes), "edgeCount": len(kg.edges)},
        }
        snapshot_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if kg.system == "lightrag":
            # E2 load mode needs the text chunks too (LightRAG's snapshot
            # round-trip only carries nodes/edges). Convert the sample
            # workdir's KV file (kv_store_text_chunks.json: {id:{value:{data:{...}}}})
            # into a flat chunks.json next to the output graph.
            kv_in = Path(kg.input_path).parent.parent / "kv_store_text_chunks.json"
            if kv_in.is_file():
                raw = json.loads(kv_in.read_text(encoding="utf-8"))
                chunks = []
                for cid, entry in (raw or {}).items():
                    data = ((entry or {}).get("value") or {}).get("data")
                    if not isinstance(data, dict):
                        data = entry if isinstance(entry, dict) else {}
                    content = data.get("content")
                    if content is None:
                        continue
                    chunks.append(
                        {
                            "id": str(cid),
                            "content": content,
                            "full_doc_id": data.get("full_doc_id") or "",
                            "file_path": data.get("file_path") or "unknown_source",
                        }
                    )
                (snapshot_path.parent / "chunks.json").write_text(
                    json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        return

    if kg.system == "hipporag":
        in_dir = Path(kg.input_path).parent
        out_dir = snapshot_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        # Carry over the sibling persistence artifacts (chunk_embeddings/,
        # fact_embeddings/ and any other working-dir files) so that
        # HippoRAG.aloadGraph restores a fully queryable index — only
        # graph.json is perturbed.
        for item in in_dir.iterdir():
            dst = out_dir / item.name
            if item.name == "graph.json" or dst.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        vertices = []
        for nid, node in kg.nodes.items():
            raw = node["raw"] if isinstance(node["raw"], dict) else {}
            if "hash_id" in raw:
                vertices.append(_with_description(raw, node["description"]))
            else:  # injected noise node: synthesize a valid vertex
                vertices.append(
                    {"hash_id": nid, "content": node["surface"], "name": nid}
                )
        by_name = {n["id"]: i for i, n in enumerate(kg.nodes.values())}
        edges = []
        for e in kg.edges:
            raw = e["raw"] if isinstance(e["raw"], dict) else {}
            edges.append(
                {
                    "source": by_name[e["source"]],
                    "target": by_name[e["target"]],
                    "weight": float(raw.get("weight", 1.0)),
                }
            )
        snapshot_path.write_text(
            json.dumps({"directed": True, "vertices": vertices, "edges": edges}, ensure_ascii=False),
            encoding="utf-8",
        )
        # Keep the entity vector store consistent with the surviving nodes
        # (HippoRAG retrieval starts from the entity KNN index, so stale
        # vectors would resurrect removed entities). Store files are
        # <working_dir>/entity_embeddings/vdb_<namespace>.json with parallel
        # hashIds/texts/embeddings arrays (EmbeddingStore.saveData).
        kept_ids = set(kg.nodes)
        for vdb_file in sorted(out_dir.glob("entity_embeddings/vdb_*.json")):
            try:
                v = json.loads(vdb_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            hash_ids = v.get("hashIds") or []
            kept = [i for i, h in enumerate(hash_ids) if h in kept_ids]
            for key, val in list(v.items()):
                if isinstance(val, list) and len(val) == len(hash_ids):
                    v[key] = [val[i] for i in kept]
            vdb_file.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
        return

    if kg.system == "youturag":
        triples = []
        for e in kg.edges:
            if isinstance(e["raw"], dict) and e["raw"].get("start_node"):
                triples.append(e["raw"])
            else:
                triples.append(
                    {
                        "start_node": {"label": e["source"], "properties": {}},
                        "relation": e["description"] or "NOISE_REL",
                        "end_node": {"label": e["target"], "properties": {}},
                    }
                )
        snapshot_path.write_text(json.dumps(triples, ensure_ascii=False), encoding="utf-8")
        return
    raise ValueError(kg.system)


def _with_description(raw: dict, description: str) -> dict:
    out = dict(raw) if isinstance(raw, dict) else {}
    out["description"] = description
    return out


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #


def apply_perturbation(
    kg: KG,
    spec: tuple[str, float | None],
    rng: random.Random,
    gold: dict | None,
) -> None:
    kind, frac = spec
    if kind == "none":
        return
    if kind == "edges":
        n = len(kg.edges)
        k = min(n, max(1, int(round(frac * n)))) if n and frac else 0
        if k:
            kg.remove_edge_indexes(set(rng.sample(range(n), k)))
    elif kind == "nodes":
        ids = list(kg.nodes)
        k = min(len(ids), max(1, int(round(frac * len(ids))))) if ids and frac else 0
        if k:
            kg.remove_nodes(set(rng.sample(ids, k)))
    elif kind == "gold_nodes":
        if not gold or not gold.get("has_gold"):
            raise SystemExit("error: P3 requires a gold row with has_gold=true")
        aliases = gold.get("gold_entity_aliases") or {}
        targets = set(gold.get("gold_entities") or [])
        for ge in gold.get("gold_edges") or []:
            for key in ("source", "target"):
                if ge.get(key):
                    targets.add(ge[key])
        victims = {
            nid
            for nid, node in kg.nodes.items()
            for t in targets
            if match_entity(t, aliases.get(t), node["surface"])
        }
        kg.remove_nodes(victims)
    elif kind == "descriptions":
        for node in kg.nodes.values():
            node["description"] = ""
        for e in kg.edges:
            e["description"] = ""
            if isinstance(e.get("raw"), dict):
                e["raw"]["description"] = ""
                if isinstance(e["raw"].get("data"), dict):
                    e["raw"]["data"]["description"] = ""
    elif kind == "noise":
        n = len(kg.nodes)
        k = max(1, int(round(frac * n)))
        base = sorted(kg.nodes)
        for i in range(k):
            nid = f"__NOISE_ENTITY_{i}"
            kg.nodes[nid] = _node(
                nid,
                f"NOISE ENTITY {i}",
                "irrelevant injected noise entity",
                {"entity_name": f"NOISE ENTITY {i}", "description": "irrelevant injected noise entity"},
            )
            anchor = rng.choice(base) if base else nid
            kg.edges.append(
                {
                    "source": nid,
                    "target": anchor,
                    "description": "NOISE_REL",
                    "raw": {"relation": "NOISE_REL"} if kg.system == "youturag" else {},
                }
            )
    else:
        raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def load_gold_row(gold_file: Path, sample_id: str) -> dict:
    from kg_common import normalize_surface

    want = normalize_surface(sample_id)
    for line in gold_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if normalize_surface(row.get("id")) == want:
            return row
    raise SystemExit(f"error: sample {sample_id!r} not found in {gold_file}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["graphrag", "lightrag", "pathrag", "hipporag", "youturag"])
    ap.add_argument("--snapshot", required=True, type=Path, help="input snapshot file (see module docstring)")
    ap.add_argument("--perturbation", required=True, choices=sorted(PERTURBATIONS))
    ap.add_argument("--out", required=True, type=Path, help="output path, same format as --snapshot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gold-file", type=Path, default=None, help="gold JSONL (required for P3)")
    ap.add_argument("--sample-id", default=None, help="sample id for P3 gold lookup")
    args = ap.parse_args(argv)

    if not args.snapshot.is_file():
        print(f"error: snapshot not found: {args.snapshot}", file=sys.stderr)
        return 2

    kg = read_kg(args.system, args.snapshot)
    kg.perturbation = args.perturbation  # type: ignore[attr-defined]
    gold = None
    if args.perturbation == "p3":
        if not args.gold_file or not args.sample_id:
            print("error: P3 needs --gold-file and --sample-id", file=sys.stderr)
            return 2
        gold = load_gold_row(args.gold_file, args.sample_id)
    rng = random.Random(args.seed)
    before = (len(kg.nodes), len(kg.edges))
    apply_perturbation(kg, PERTURBATIONS[args.perturbation], rng, gold)
    write_kg(kg, args.out)
    after = (len(kg.nodes), len(kg.edges))
    print(
        f"[perturb] {args.system} {args.perturbation}: "
        f"nodes {before[0]}->{after[0]}, edges {before[1]}->{after[1]} "
        f"(seed={args.seed}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
