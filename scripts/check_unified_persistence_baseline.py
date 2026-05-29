#!/usr/bin/env python3
"""Validate Phase P0 unified persistence baseline inventory and fixture locks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_RAG_IDS = {
    "GRAPH_RAG",
    "LIGHT_RAG",
    "PATH_RAG",
    "HIPPO_RAG",
    "CAUSAL_RAG",
    "CAUSAL_HIPPO_RAG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="src/test/resources/unified_persistence_parity/p0",
        help="Root directory containing inventory.json and fixture_hashes.json",
    )
    return parser.parse_args()


def sha256_hex(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_inventory(root: Path) -> list[str]:
    problems: list[str] = []
    inventory_path = root / "inventory.json"
    if not inventory_path.exists():
        return [f"Missing inventory file: {inventory_path}"]

    inventory = load_json(inventory_path)
    rows = inventory.get("rags")
    if not isinstance(rows, list):
        return ["inventory.json must contain a 'rags' array"]

    rag_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            problems.append("Each inventory row must be an object")
            continue
        rag_id = row.get("ragId")
        if not isinstance(rag_id, str):
            problems.append("Each inventory row must include string ragId")
            continue
        rag_ids.add(rag_id)

        baseline = row.get("baseline")
        if not isinstance(baseline, dict):
            problems.append(f"{rag_id}: missing baseline object")
            continue

        inspect_fixture = baseline.get("inspectFixture")
        if not isinstance(inspect_fixture, str) or not inspect_fixture.strip():
            problems.append(f"{rag_id}: missing baseline.inspectFixture")
        else:
            inspect_path = root / inspect_fixture
            if not inspect_path.exists():
                problems.append(f"{rag_id}: inspect fixture missing: {inspect_path}")

        status = baseline.get("roundTripFixtureStatus")
        if status == "available":
            fixture_root = baseline.get("roundTripFixtureRoot")
            if not isinstance(fixture_root, str) or not fixture_root.strip():
                problems.append(f"{rag_id}: available round-trip fixture must define baseline.roundTripFixtureRoot")
            else:
                fixture_path = root / fixture_root
                if not fixture_path.exists():
                    problems.append(f"{rag_id}: round-trip fixture root missing: {fixture_path}")

    missing = EXPECTED_RAG_IDS - rag_ids
    extra = rag_ids - EXPECTED_RAG_IDS
    if missing:
        problems.append(f"Missing rag IDs in inventory: {sorted(missing)}")
    if extra:
        problems.append(f"Unexpected rag IDs in inventory: {sorted(extra)}")
    return problems


def validate_hash_lock(root: Path) -> list[str]:
    problems: list[str] = []
    lock_path = root / "fixture_hashes.json"
    if not lock_path.exists():
        return [f"Missing lock file: {lock_path}"]

    lock = load_json(lock_path)
    files = lock.get("files")
    if not isinstance(files, dict):
        return ["fixture_hashes.json must contain an object field named 'files'"]

    for rel_path, expected_hash in sorted(files.items()):
        if not isinstance(rel_path, str) or not isinstance(expected_hash, str):
            problems.append("fixture_hashes.json entries must be string:string")
            continue
        full_path = root / rel_path
        if not full_path.exists():
            problems.append(f"Locked fixture file missing: {full_path}")
            continue
        actual_hash = sha256_hex(full_path)
        if actual_hash != expected_hash:
            problems.append(
                f"Hash mismatch for {rel_path}: expected={expected_hash} actual={actual_hash}",
            )
    return problems


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    problems = []
    problems.extend(validate_inventory(root))
    problems.extend(validate_hash_lock(root))

    if problems:
        print("Unified persistence P0 baseline validation failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Unified persistence P0 baseline validation passed.")
    print(f"Root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
