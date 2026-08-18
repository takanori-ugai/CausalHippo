"""Generate the SWO69 30-sample evaluation sets (T0).

The original 300-sample experiment files exist under ``data/``, but the raw
upstream sources (``MuSiQue/data/*.jsonl``, ``Webis/...``) are not present on
this machine, so the 30-sample sets are derived *deterministically* from the
300-sample files (seed 42), stratified to preserve the balance guarantees:

- MuSiQue: by hop count (2/3/4 hops -> 10/10/10)
- Causal-Reasoning-QA: by ``metadata.label_true`` (15/15)
- Webis: by ``metadata.category`` (proportional allocation)

Outputs (names fixed by SWO69/SWO_PLAN.md §2.2):
  data/musique_experiment/musique_dev_balanced_30.jsonl
  data/causal_experiment/causal_qa_balanced_30.jsonl
  data/webis_experiment/webis_train_balanced_30.jsonl

Re-running the script is idempotent (same seed -> same 30 rows).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
TARGET = 30

DATASETS = {
    "musique": {
        "src": ROOT / "data/musique_experiment/musique_dev_balanced_300.jsonl",
        "dst": ROOT / "data/musique_experiment/musique_dev_balanced_30.jsonl",
        "strata_key": lambda r: len(r.get("question_decomposition", [])),
    },
    "causal": {
        "src": ROOT / "data/causal_experiment/causal_qa_balanced_300.jsonl",
        "dst": ROOT / "data/causal_experiment/causal_qa_balanced_30.jsonl",
        "strata_key": lambda r: str(r.get("metadata", {}).get("label_true")),
    },
    "webis": {
        "src": ROOT / "data/webis_experiment/webis_train_balanced_300.jsonl",
        "dst": ROOT / "data/webis_experiment/webis_train_balanced_30.jsonl",
        "strata_key": lambda r: str(r.get("metadata", {}).get("category")),
    },
}


def proportional_quotas(group_sizes: dict, target: int) -> dict:
    """Largest-remainder allocation of `target` across groups."""
    total = sum(group_sizes.values())
    raw = {k: target * v / total for k, v in group_sizes.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    remainder = target - sum(quotas.values())
    for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if remainder <= 0:
            break
        quotas[k] += 1
        remainder -= 1
    return quotas


def pick(dataset: str, spec: dict) -> list[dict]:
    rows = [json.loads(line) for line in spec["src"].read_text().splitlines() if line.strip()]
    groups: dict = defaultdict(list)
    for row in rows:
        groups[spec["strata_key"](row)].append(row)

    quotas = proportional_quotas({k: len(v) for k, v in groups.items()}, TARGET)
    rng = random.Random(f"{dataset}-{SEED}")
    selected: list[dict] = []
    for key in sorted(groups, key=str):
        bucket = groups[key][:]
        rng.shuffle(bucket)
        selected.extend(bucket[: quotas[key]])
    rng.shuffle(selected)
    return selected


def main() -> None:
    for dataset, spec in DATASETS.items():
        if not spec["src"].exists():
            raise SystemExit(f"missing source file: {spec['src']}")
        picked = pick(dataset, spec)
        if len(picked) != TARGET:
            raise SystemExit(f"{dataset}: expected {TARGET} rows, got {len(picked)}")

        src_ids = {json.loads(l)["id"] for l in spec["src"].read_text().splitlines() if l.strip()}
        missing = [r["id"] for r in picked if r["id"] not in src_ids]
        if missing:
            raise SystemExit(f"{dataset}: picked rows not in 300-sample source: {missing[:5]}")

        spec["dst"].parent.mkdir(parents=True, exist_ok=True)
        with spec["dst"].open("w") as fh:
            for row in picked:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        strata = Counter(spec["strata_key"](r) for r in picked)
        print(
            f"{dataset}: wrote {spec['dst'].relative_to(ROOT)} "
            f"(n={len(picked)}, strata={dict(sorted(strata.items(), key=lambda kv: str(kv[0])))})"
        )


if __name__ == "__main__":
    main()
