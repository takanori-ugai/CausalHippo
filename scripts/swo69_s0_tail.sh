#!/usr/bin/env bash
# SWO69 S0 tail: re-run the two remaining smoke steps (6, 7) with more
# retry budget than the original chain — the first attempt died on GPU
# contention with the 27B session model (6 × 60s Ollama timeouts).
# Steps already producing non-empty per_question rows are skipped.
set -uo pipefail

cd "$(dirname "$0")/.."

MUSIQUE=data/musique_experiment/musique_dev_balanced_30.jsonl
BASE_CONFIG=config/common_rag_ollama.json
OUT_ROOT=eval_results/swo69_s0
GRAPH_MAIN=shared.eval.graph.MultiConditionGraphExperimentKt

# PathRAG/LightRAG resolve the chat model from the config JSON (shared.modelName),
# ignoring --llm-model — so generate a per-model config for each step.
# Their workingDir is ALSO config-driven (pathrag.workingDir / lightrag.workingDir);
# if two steps share the configured dir and the same sample id (S0 uses --limit 1
# on the same first sample), the later step's resetDirectory() destroys the
# earlier step's kg_snapshot. Pin each step's storage under its own output dir.
mkcfg() { # model storage_root -> config path
  local model="$1"
  local storage="$2"
  local safe="${model//:/_}"
  local h; h=$(printf '%s' "$storage" | md5sum | cut -c1-8)
  local out="config/swo69_cfg_${safe}_${h}.json"
  if [[ ! -f "$out" ]]; then
    python3 - "$BASE_CONFIG" "$out" "$model" "$storage" <<'EOF'
import json, sys
src, dst, model, storage = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(src, encoding="utf-8") as fh:
    cfg = json.load(fh)
cfg.setdefault("shared", {})["modelName"] = model
cfg.setdefault("pathrag", {})["workingDir"] = storage + "/pathrag"
cfg.setdefault("lightrag", {})["workingDir"] = storage + "/lightrag"
with open(dst, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
EOF
  fi
  echo "$out"
}

export LLM_PROVIDER=ollama
export OPENAI_API_KEY=dummy
export OPENAI_EMBEDDING_MODEL=nomic-embed-text
export OPENAI_API_BASE=http://127.0.0.1:11434
export EMBED_PROVIDER=ollama
export OLLAMA_RETRY_ATTEMPTS=12
export OLLAMA_RETRY_BACKOFF_MS=5000
export OPENAI_RETRY_ATTEMPTS=12
export OPENAI_RETRY_BACKOFF_MS=5000

LOG="$OUT_ROOT/S0_tail.log"

already_done() {
  local outdir="$1"; shift
  local c
  for c in "$@"; do
    [[ -s "$outdir/per_question/$c.jsonl" ]] || return 1
  done
  return 0
}

step() {
  local name="$1"; local outdir="$2"; local cond="$3"; shift 3
  if already_done "$outdir" "$cond"; then
    echo "=== [S0tail] $(date -u +%H:%M:%S) SKIP  $name (output exists) ===" | tee -a "$LOG"
    return 0
  fi
  echo "=== [S0tail] $(date -u +%H:%M:%S) START $name ===" | tee -a "$LOG"
  local t0 t1
  t0=$(date +%s)
  "$@" > "$OUT_ROOT/$name.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  echo "=== [S0tail] $(date -u +%H:%M:%S) END   $name rc=$rc wall=$((t1 - t0))s ===" | tee -a "$LOG"
  return $rc
}

mkdir -p "$OUT_ROOT"

# 6. Weak-model saturation check (llama3.2:3b, PathRAG single-pass JSON).
#    NOTE: model comes from the config file (PathRAG ignores --llm-model).
step pathrag_llama3b "$OUT_ROOT/pathrag_llama3b" "pathrag" \
  scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions pathrag \
    --config "$(mkcfg llama3.2:3b "$OUT_ROOT/pathrag_llama3b/storage")" \
    --top-k 5 --parallelism 1 --limit 1 \
    --output-dir "$OUT_ROOT/pathrag_llama3b"

# 7. 27B throughput reference (the E0 extraction/generation model).
step pathrag_qwen27b "$OUT_ROOT/pathrag_qwen27b" "pathrag" \
  scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions pathrag \
    --config "$(mkcfg qwen3.8:27b "$OUT_ROOT/pathrag_qwen27b/storage")" \
    --top-k 5 --parallelism 1 --limit 1 \
    --output-dir "$OUT_ROOT/pathrag_qwen27b"

# 8. GraphRAG pipeline smoke (strong model; non-unified runner, as in T6 E0).
#    OPENAI_API_BASE must carry the /v1 OpenAI-compatible prefix for Ollama;
#    OPENAI_MODEL selects the extraction model (graphrag reads env, not config).
step graphrag_qwen27b "$OUT_ROOT/graphrag_ollama" "graphrag" \
  env OPENAI_API_BASE=http://127.0.0.1:11434/v1 OPENAI_MODEL=qwen3.8:27b \
  scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions graphrag \
    --provider openai --llm-base-url http://127.0.0.1:11434/v1 \
    --llm-model qwen3.8:27b \
    --config "$(mkcfg qwen3.8:27b "$OUT_ROOT/graphrag_ollama/storage")" \
    --top-k 5 --parallelism 1 --limit 1 \
    --output-dir "$OUT_ROOT/graphrag_ollama"

echo "=== [S0tail] $(date -u +%H:%M:%S) all tail smoke steps finished ===" | tee -a "$LOG"
