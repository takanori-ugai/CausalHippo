#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/musique_experiment/musique_dev_balanced_300.jsonl"
OUTPUT_DIR=""
MANIFEST_PATH=""
CONDITIONS="all"
TOP_K="5"
PARALLELISM="2"
LIMIT=""
LLM_MODEL="${LLM_MODEL:-gpt-5.4-mini}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
PROVIDER="${LLM_PROVIDER:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-}"
SKIP_AGGREGATE="false"

usage() {
  cat <<'EOF_USAGE'
Usage: ./run_experiment_graph.sh [options]

Options:
  --data <path>            Input MuSiQue JSONL
  --output-dir <path>      Output directory (default: eval_results/graph_multicondition_<timestamp>)
  --manifest <path>        Manifest JSONL (optional; auto-detected when omitted)
  --conditions <list>      all or comma list of condition IDs
  --top-k <int>            Retrieval topK (default: 5)
  --parallelism <int>      Number of samples to execute in parallel (default: 2)
  --limit <int>            Limit samples (optional)
  --llm-model <name>       Override generation model
  --embedding-model <name> Override embedding model
  --provider <name>        openai|ollama (default: env LLM_PROVIDER or openai)
  --llm-base-url <url>     Optional base URL
  --skip-aggregate         Skip CSV aggregation step
  -h, --help               Show this help

Conditions:
  all (lightrag,pathrag,graphrag)
  lightrag
  pathrag
  graphrag

Example:
  ./run_experiment_graph.sh \
    --data data/musique_experiment/musique_dev_balanced_300.jsonl \
    --conditions all \
    --top-k 5
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST_PATH="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --llm-base-url) LLM_BASE_URL="$2"; shift 2 ;;
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
  OUTPUT_DIR="eval_results/graph_multicondition_$(date -u +%Y%m%d_%H%M%S)"
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Data file not found: $DATA_PATH" >&2
  exit 1
fi

if [[ "$CONDITIONS" == "all" || ",$CONDITIONS," == *",graphrag,"* ]]; then
  if [[ "$PROVIDER" != "openai" ]]; then
    echo "GraphRAG requires --provider openai (current: $PROVIDER)" >&2
    exit 1
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is required when running graphrag condition." >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"

APP_ARGS=(
  --data "$DATA_PATH"
  --output-dir "$OUTPUT_DIR"
  --conditions "$CONDITIONS"
  --top-k "$TOP_K"
  --parallelism "$PARALLELISM"
  --provider "$PROVIDER"
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

echo "[run_experiment_graph] output: $OUTPUT_DIR"
echo "[run_experiment_graph] conditions: $CONDITIONS"
echo "[run_experiment_graph] QA metrics: exact_match, precision, recall, f1"
echo "[run_experiment_graph] BertScore metrics: bertscore_precision, bertscore_recall, bertscore_f1"

# PathRAG reads these names directly from env.
export LLM_PROVIDER="$PROVIDER"
export OPENAI_MODEL="$LLM_MODEL"
export OPENAI_EMBEDDING_MODEL="$EMBEDDING_MODEL"
if [[ -n "$LLM_BASE_URL" ]]; then
  export OPENAI_API_BASE="$LLM_BASE_URL"
fi

./gradlew --quiet execute \
  -PmainClass=causalrag.examples.graph.MultiConditionGraphExperimentKt \
  --args="${APP_ARGS[*]}"

if [[ "$SKIP_AGGREGATE" == "false" ]]; then
  python3 scripts/aggregate_experiment_results.py --input-dir "$OUTPUT_DIR"
fi

echo "[run_experiment_graph] done"
echo "[run_experiment_graph] per-question JSONL: $OUTPUT_DIR/per_question"
echo "[run_experiment_graph] summary CSV: $OUTPUT_DIR/summary_by_condition.csv"
echo "[run_experiment_graph] summary by category CSV: $OUTPUT_DIR/summary_by_condition_category.csv"
echo "[run_experiment_graph] label-aware summary CSV: $OUTPUT_DIR/summary_label_aware_by_condition.csv"
echo "[run_experiment_graph] per-question CSV: $OUTPUT_DIR/per_question_metrics.csv"
echo "[run_experiment_graph] note: CSV outputs include BertScore columns"
