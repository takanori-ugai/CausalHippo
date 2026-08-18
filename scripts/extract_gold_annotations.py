"""T2 (SWO69): extract gold annotations for KG-quality evaluation.

Reads the 30-sample SWO69 datasets and emits, per dataset, a JSONL file with
the structural gold used by T1/T3 (§3.1 of SWO69/SWO_PLAN.md):

- MuSiQue:        gold entity set = {qd[i].answer} ∪ {answer} ∪ answer_aliases;
                  gold chain = ordered sub-question answers (bridge entities
                  plus the final answer, which equals `answer`).
- Causal-Reasoning-QA: gold entities = {cause_candidate, effect};
                  gold edge (cause_candidate -> effect) with `label_true`.
- Webis:          no structural gold (answer is free text); emitted with
                  `has_gold = false` so downstream joins stay uniform.

Usage:
  python3 scripts/extract_gold_annotations.py            # all three datasets
  python3 scripts/extract_gold_annotations.py --datasets musique,causal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from kg_common import normalize_surface  # noqa: E402  (scripts/ on sys.path)

DATASETS = {
    "musique": ROOT / "data/musique_experiment/musique_dev_balanced_30.jsonl",
    "causal": ROOT / "data/causal_experiment/causal_qa_balanced_30.jsonl",
    "webis": ROOT / "data/webis_experiment/webis_train_balanced_30.jsonl",
}

OUT_DIR = ROOT / "data/gold_annotations"


def gold_musique(record: dict) -> dict:
    qd = record.get("question_decomposition", [])
    chain = [q.get("answer", "") for q in qd if q.get("answer")]
    final_answer = record.get("answer", "")
    if not chain or chain[-1] != final_answer:
        # Robustness: the final answer is the terminal chain element.
        if final_answer:
            chain = (chain or []) + [final_answer]
    aliases = record.get("answer_aliases", []) or []
    gold_entities = list(dict.fromkeys([*chain, final_answer, *aliases]))
    return {
        "has_gold": True,
        "final_answer": final_answer,
        "gold_entities": gold_entities,
        "gold_entity_aliases": {final_answer: aliases},
        "gold_chain": chain,
        "gold_edges": [],
    }


def gold_causal(record: dict) -> dict:
    meta = record.get("metadata", {})
    cause = meta.get("cause_candidate", "")
    effect = meta.get("effect", "")
    label_true = meta.get("label_true")
    return {
        "has_gold": True,
        "final_answer": record.get("answer", ""),
        "gold_entities": [e for e in (cause, effect) if e],
        "gold_entity_aliases": {},
        "gold_chain": [],  # causal dataset: no ordered multi-hop chain
        "gold_edges": (
            [{"source": cause, "target": effect, "label_true": bool(label_true)}]
            if cause and effect
            else []
        ),
        "label_true": bool(label_true),
    }


def gold_webis(record: dict) -> dict:
    return {
        "has_gold": False,
        "final_answer": record.get("answer", ""),
        "gold_entities": [],
        "gold_entity_aliases": {},
        "gold_chain": [],
        "gold_edges": [],
    }


BUILDERS = {"musique": gold_musique, "causal": gold_causal, "webis": gold_webis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="musique,causal,webis",
        help="Comma list of: musique,causal,webis",
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset '{name}' (choices: {','.join(DATASETS)})")
        src = DATASETS[name]
        builder = BUILDERS[name]
        records = [json.loads(line) for line in src.read_text().splitlines() if line.strip()]

        out_path = out_dir / f"{name}_gold_30.jsonl"
        n_gold = 0
        with out_path.open("w") as fh:
            for rec in records:
                gold = builder(rec)
                row = {
                    "id": rec["id"],
                    "dataset": name,
                    "question": rec.get("question", ""),
                    **gold,
                }
                # Convenience: normalized gold surfaces for T1/T3 matching.
                row["gold_entities_norm"] = [normalize_surface(e) for e in row["gold_entities"] if e]
                n_gold += int(row["has_gold"])
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{name}: wrote {out_path.relative_to(ROOT)} (n={len(records)}, with_gold={n_gold})")


if __name__ == "__main__":
    main()
