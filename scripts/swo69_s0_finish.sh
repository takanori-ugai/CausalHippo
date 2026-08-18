#!/usr/bin/env bash
# SWO69 S0 smoke chain (full): 1-sample runs for every system path, per
# SWO_PLAN.md §8 M1. Runs sequentially in the background; each step is
# skipped when its per_question output already exists and is non-empty.
#
# Offline: uses scripts/run_kt.sh (prebuilt shadow jar) instead of gradlew.
set -uo pipefail

cd "$(dirname "$0")/.."

MUSIQUE=data/musique_experiment/musique_dev_balanced_30.jsonl
WEBIS=data/webis_experiment/webis_train_balanced_30.jsonl
CONFIG=config/common_rag_ollama.json
OUT_ROOT=eval_results/swo69_s0
COMMON="--top-k 5 --parallelism 1 --limit 1 --config $CONFIG"
GRAPH_MAIN=shared.eval.graph.MultiConditionGraphExperimentKt
RAG_MAIN=shared.eval.MultiConditionExperimentKt
# PathRAG reads these names directly from env (see run_experiment_graph.sh).
# OPENAI_API_KEY must be non-empty or PathRAG falls back to STUBBED embeddings.
# EMBED_PROVIDER=ollama makes PathRAG use OllamaEmbeddingModel with the correct
# nomic-embed-text dim (768) instead of the 1536-dim OpenAI-path guess.
# The langchain4j Ollama client has a 60s request timeout; under GPU contention
# with the 27B session model, raise retries so calls get a chance to finish.
export LLM_PROVIDER=ollama
export OPENAI_API_KEY=dummy
export OPENAI_EMBEDDING_MODEL=nomic-embed-text
export OPENAI_API_BASE=http://127.0.0.1:11434
export EMBED_PROVIDER=ollama
export OLLAMA_RETRY_ATTEMPTS=6
export OLLAMA_RETRY_BACKOFF_MS=2000
export OPENAI_RETRY_ATTEMPTS=6
export OPENAI_RETRY_BACKOFF_MS=2000

LOG="$OUT_ROOT/S0_finish.log"

# already_done <out-dir> <condition...> : true when every condition has rows
already_done() {
  local outdir="$1"; shift
  local c
  for c in "$@"; do
    [[ -s "$outdir/per_question/$c.jsonl" ]] || return 1
  done
  return 0
}

step() {
  local name="$1"; local outdir="$2"; local conds_csv="$3"; shift 3
  local IFS=','
  local conds=($conds_csv)
  if already_done "$outdir" "${conds[@]}"; then
    echo "=== [S0] $(date -u +%H:%M:%S) SKIP  $name (output exists) ===" | tee -a "$LOG"
    return 0
  fi
  echo "=== [S0] $(date -u +%H:%M:%S) START $name ===" | tee -a "$LOG"
  local t0 t1
  t0=$(date +%s)
  "$@" > "$OUT_ROOT/$name.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  echo "=== [S0] $(date -u +%H:%M:%S) END   $name rc=$rc wall=$((t1 - t0))s ===" | tee -a "$LOG"
  return $rc
}

mkdir -p "$OUT_ROOT"

# 1. LightRAG + PathRAG pipeline sanity (fast mid model).
step lightrag_pathrag_gemma "$OUT_ROOT/lightrag_pathrag_gemma" "lightrag,pathrag" \
  scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions lightrag,pathrag \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/lightrag_pathrag_gemma"

# 2. GraphRAG through Ollama's OpenAI-compatible endpoint.
# NOTE: the index pipeline (Workflows.kt) builds its chat/embedding clients from
# env, not CLI args: OPENAI_API_BASE must point at the /v1 OpenAI-compatible
# endpoint and OPENAI_MODEL selects the extraction model (gpt-4o-mini otherwise).
# Requires the rebuilt jar (see SWO69/SWO_PLAN.md T7 rebuild note).
step graphrag_ollama "$OUT_ROOT/graphrag_ollama" "graphrag" \
  env OPENAI_API_KEY=dummy \
     OPENAI_API_BASE=http://127.0.0.1:11434/v1 \
     OPENAI_MODEL=gemma4:e4b \
     OPENAI_EMBEDDING_MODEL=nomic-embed-text \
     scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions graphrag \
    --provider openai --llm-base-url http://127.0.0.1:11434/v1 \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/graphrag_ollama"

# 3. YoutuRAG (unified API required).
step youturag_gemma "$OUT_ROOT/youturag_gemma" "youturag" \
  scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions youturag --use-unified-api \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/youturag_gemma"

# 4. HippoRAG (RAG runner).
step hipporag_gemma "$OUT_ROOT/hipporag_gemma" "hipporag_graph" \
  scripts/run_kt.sh "$RAG_MAIN" --data "$MUSIQUE" --conditions hipporag_graph \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/hipporag_gemma"

# 5. Untested Webis execution path (HippoRAG runner on Webis JSONL).
step webis_hipporag_gemma "$OUT_ROOT/webis_hipporag_gemma" "hipporag_graph" \
  scripts/run_kt.sh "$RAG_MAIN" --data "$WEBIS" --conditions hipporag_graph \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/webis_hipporag_gemma"

# 6. Weak-model saturation check (llama3.2:3b, PathRAG single-pass JSON).
step pathrag_llama3b "$OUT_ROOT/pathrag_llama3b" "pathrag" \
  env OPENAI_MODEL=llama3.2:3b scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions pathrag \
    --llm-model llama3.2:3b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/pathrag_llama3b"

# 7. 27B throughput reference (the E0 extraction/generation model).
step pathrag_qwen27b "$OUT_ROOT/pathrag_qwen27b" "pathrag" \
  env OPENAI_MODEL=qwen3.8:27b scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions pathrag \
    --llm-model qwen3.8:27b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/pathrag_qwen27b"

echo "=== [S0] $(date -u +%H:%M:%S) all smoke steps finished ===" | tee -a "$LOG"
