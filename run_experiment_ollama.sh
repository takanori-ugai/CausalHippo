#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/musique_experiment/musique_dev_balanced_300.jsonl"
OUTPUT_DIR=""
CONFIG_PATH="config/common_rag_ollama.json"
MANIFEST_PATH=""
CONDITIONS="all"
TOP_K="5"
PARALLELISM="5"
LIMIT=""
LLM_MODEL="${LLM_MODEL:-gemma4:e4b}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-embed-text}"
PROVIDER="${LLM_PROVIDER:-ollama}"
LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
TEMPLATE_STYLE="detailed_musique"
SKIP_AGGREGATE="false"
USE_UNIFIED_API="false"
USE_UNIFIED_PERSISTENCE="false"

usage() {
  cat <<'EOF'
Usage: ./run_experiment_ollama.sh [options]

Options:
  --data <path>            Input MuSiQue JSONL
  --output-dir <path>      Output directory (default: eval_results/multicondition_<timestamp>)
  --config <path>          Pipeline/common config path (default: config/common_rag_ollama.json)
  --manifest <path>        Manifest JSONL (optional; auto-detected when omitted)
  --conditions <list>      all or comma list of condition IDs
  --top-k <int>            Retrieval topK (default: 5)
  --parallelism <int>      Number of samples to execute in parallel (default: 5)
  --limit <int>            Limit samples (optional)
  --llm-model <name>       Override generation model (default: qwen3.8:27b)
  --embedding-model <name> Override embedding model (default: nomic-embed-text)
  --provider <name>        openai|azure|ollama|gemini (default: env LLM_PROVIDER or ollama)
  --llm-base-url <url>     Optional base URL (default: http://127.0.0.1:11434)
  --template-style <name>  Prompt template style (default: detailed_musique)
  --use-unified-api        Route conditions through shared unified adapters
  --use-unified-persistence Enable unified persistence SPI sidecar in unified mode
  --skip-aggregate         Skip CSV aggregation step
  -h, --help               Show this help

Example:
  ./run_experiment_ollama.sh \
    --data data/musique_experiment/musique_dev_balanced_300.jsonl \
    --conditions all \
    --top-k 5

Notes:
  Uses a local Ollama server (default http://127.0.0.1:11434).
  NOTE: use 127.0.0.1, not localhost — this box resolves localhost to ::1
  first and Ollama only listens on the IPv4 loopback, so GraphRAG's HTTP
  client (no IPv4 fallback) connect-times-out otherwise.
  Make sure both models are pulled first, e.g.:
    ollama pull qwen3.8:27b
    ollama pull nomic-embed-text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --manifest) MANIFEST_PATH="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --llm-base-url) LLM_BASE_URL="$2"; shift 2 ;;
    --template-style) TEMPLATE_STYLE="$2"; shift 2 ;;
    --use-unified-api) USE_UNIFIED_API="true"; shift 1 ;;
    --use-unified-persistence) USE_UNIFIED_PERSISTENCE="true"; shift 1 ;;
    --skip-aggregate) SKIP_AGGREGATE="true"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="eval_results/multicondition_ollama_$(date -u +%Y%m%d_%H%M%S)"
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Data file not found: $DATA_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

APP_ARGS=(
  --data "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --config "$CONFIG_PATH"
  --conditions "$CONDITIONS"
  --top-k "$TOP_K"
  --parallelism "$PARALLELISM"
  --provider "$PROVIDER"
  --template-style "$TEMPLATE_STYLE"
  --use-unified-api "$USE_UNIFIED_API"
  --use-unified-persistence "$USE_UNIFIED_PERSISTENCE"
)

if [[ -n "$MANIFEST_PATH" ]]; then
  APP_ARGS+=(--manifest "$MANIFEST_PATH")
fi

if [[ -n "$LIMIT" ]]; then
  APP_ARGS+=(--limit "$LIMIT")
fi

if [[ -n "$LLM_MODEL" ]]; then
  APP_ARGS+=(--llm-model "$LLM_MODEL")
fi

if [[ -n "$EMBEDDING_MODEL" ]]; then
  APP_ARGS+=(--embedding-model "$EMBEDDING_MODEL")
fi

if [[ -n "$LLM_BASE_URL" ]]; then
  APP_ARGS+=(--llm-base-url "$LLM_BASE_URL")
fi

echo "[run_experiment_ollama] output: $OUTPUT_DIR"
echo "[run_experiment_ollama] provider: $PROVIDER"
echo "[run_experiment_ollama] llm model: $LLM_MODEL"
echo "[run_experiment_ollama] embedding model: $EMBEDDING_MODEL"
echo "[run_experiment_ollama] base url: $LLM_BASE_URL"
echo "[run_experiment_ollama] conditions: $CONDITIONS"
echo "[run_experiment_ollama] QA metrics: exact_match, precision, recall, f1"
echo "[run_experiment_ollama] BertScore metrics: bertscore_precision, bertscore_recall, bertscore_f1"

#./gradlew --quiet execute \
#  -PmainClass=shared.eval.MultiConditionExperimentKt \
#  --args="${APP_ARGS[*]}"
echo ${APP_ARGS[*]}
java -cp build/libs/causalrag-0.0.1-all.jar shared.eval.MultiConditionExperimentKt ${APP_ARGS[*]}

if [[ "$SKIP_AGGREGATE" == "false" ]]; then
  python3 scripts/aggregate_experiment_results.py --input-dir "$OUTPUT_DIR"
fi

echo "[run_experiment_ollama] done"
echo "[run_experiment_ollama] per-question JSONL: $OUTPUT_DIR/per_question"
echo "[run_experiment_ollama] summary CSV: $OUTPUT_DIR/summary_by_condition.csv"
echo "[run_experiment_ollama] summary by category CSV: $OUTPUT_DIR/summary_by_condition_category.csv"
echo "[run_experiment_ollama] label-aware summary CSV: $OUTPUT_DIR/summary_label_aware_by_condition.csv"
echo "[run_experiment_ollama] per-question CSV: $OUTPUT_DIR/per_question_metrics.csv"
echo "[run_experiment_ollama] note: CSV outputs include BertScore columns"
