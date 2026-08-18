#!/usr/bin/env bash
# SWO69 S0 re-run: the six smoke steps that previously failed under ./gradlew
# (offline box cannot download the Gradle distribution). Re-executed through
# scripts/run_kt.sh (prebuilt shadow jar). Step 1 (lightrag+pathrag, gemma)
# already passed and is skipped.
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
export LLM_PROVIDER=ollama
export OPENAI_EMBEDDING_MODEL=nomic-embed-text
export OPENAI_API_BASE=http://127.0.0.1:11434

step() {
  local name="$1"; shift
  echo "=== [S0] $(date -u +%H:%M:%S) START $name ===" | tee -a "$OUT_ROOT/S0_run.log"
  local t0 t1
  t0=$(date +%s)
  "$@" > "$OUT_ROOT/$name.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  echo "=== [S0] $(date -u +%H:%M:%S) END   $name rc=$rc wall=$((t1 - t0))s (log: $OUT_ROOT/$name.log) ===" | tee -a "$OUT_ROOT/S0_run.log"
}

mkdir -p "$OUT_ROOT"

# 2. GraphRAG through Ollama's OpenAI-compatible endpoint.
# Index pipeline reads OPENAI_API_BASE (must include /v1) + OPENAI_MODEL from env.
step graphrag_ollama \
  env OPENAI_API_KEY=dummy \
     OPENAI_API_BASE=http://127.0.0.1:11434/v1 \
     OPENAI_MODEL=gemma4:e4b \
     OPENAI_EMBEDDING_MODEL=nomic-embed-text \
     scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions graphrag \
    --provider openai --llm-base-url http://127.0.0.1:11434/v1 \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/graphrag_ollama"

# 3. YoutuRAG (unified API required).
step youturag_gemma \
  scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions youturag --use-unified-api \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/youturag_gemma"

# 4. HippoRAG (RAG runner).
step hipporag_gemma \
  scripts/run_kt.sh "$RAG_MAIN" --data "$MUSIQUE" --conditions hipporag_graph \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/hipporag_gemma"

# 5. Untested Webis execution path (HippoRAG runner on Webis JSONL).
step webis_hipporag_gemma \
  scripts/run_kt.sh "$RAG_MAIN" --data "$WEBIS" --conditions hipporag_graph \
    --llm-model gemma4:e4b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/webis_hipporag_gemma"

# 6. Weak-model saturation check (llama3.2:3b, PathRAG single-pass JSON).
step pathrag_llama3b \
  env OPENAI_MODEL=llama3.2:3b scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions pathrag \
    --llm-model llama3.2:3b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/pathrag_llama3b"

# 7. 27B throughput reference (the E0 extraction/generation model).
step pathrag_qwen27b \
  env OPENAI_MODEL=qwen3.8:27b scripts/run_kt.sh "$GRAPH_MAIN" --data "$MUSIQUE" --conditions pathrag \
    --llm-model qwen3.8:27b --embedding-model nomic-embed-text \
    $COMMON --output-dir "$OUT_ROOT/pathrag_qwen27b"

echo "=== [S0] $(date -u +%H:%M:%S) all smoke steps finished ===" | tee -a "$OUT_ROOT/S0_run.log"
