#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="data/causal_experiment/causal_qa_balanced_300.jsonl"
OUTPUT_DIR=""
CONFIG_PATH="config/common_rag.json"
CONDITIONS="all"
TOP_K="5"
PARALLELISM="5"
LIMIT=""
LLM_MODEL="${LLM_MODEL:-gpt-5.4-mini}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-}"
PROVIDER="${LLM_PROVIDER:-openai}"
LLM_BASE_URL="${LLM_BASE_URL:-}"
TEMPLATE_STYLE="experiments"
SKIP_AGGREGATE="false"

usage() {
  cat <<'EOF_USAGE'
Usage: ./run_experiment_causal.sh [options]

Options:
  --data <path>            Input MuSiQue-style causal QA JSONL (default: data/causal_experiment/causal_qa_balanced_300.jsonl)
  --output-dir <path>      Output directory (default: eval_results/causal_multicondition_<timestamp>)
  --config <path>          Pipeline/common config path (default: config/common_rag.json)
  --conditions <list>      all or comma list of condition IDs
  --top-k <int>            Retrieval topK (default: 5)
  --parallelism <int>      Number of samples to execute in parallel (default: 5)
  --limit <int>            Limit samples (optional)
  --llm-model <name>       Override generation model
  --embedding-model <name> Override embedding model
  --provider <name>        openai|azure|ollama (default: env LLM_PROVIDER or openai)
  --llm-base-url <url>     Optional base URL
  --template-style <name>  Prompt template style (default: experiments)
  --skip-aggregate         Skip CSV aggregation step
  -h, --help               Show this help

Example:
  ./run_experiment_causal.sh \
    --data data/causal_experiment/causal_qa_balanced_300.jsonl \
    --conditions all \
    --top-k 5
EOF_USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data) DATA_PATH="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    --top-k) TOP_K="$2"; shift 2 ;;
    --parallelism) PARALLELISM="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --llm-base-url) LLM_BASE_URL="$2"; shift 2 ;;
    --template-style) TEMPLATE_STYLE="$2"; shift 2 ;;
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
  OUTPUT_DIR="eval_results/causal_multicondition_$(date -u +%Y%m%d_%H%M%S)"
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
)

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

echo "[run_experiment_causal] output: $OUTPUT_DIR"
echo "[run_experiment_causal] conditions: $CONDITIONS"
echo "[run_experiment_causal] QA metrics: exact_match, precision, recall, f1"
echo "[run_experiment_causal] BertScore metrics: bertscore_precision, bertscore_recall, bertscore_f1"

./gradlew --quiet execute \
  -PmainClass=shared.eval.MultiConditionExperimentKt \
  --args="${APP_ARGS[*]}"

if [[ "$SKIP_AGGREGATE" == "false" ]]; then
  python3 scripts/aggregate_experiment_results.py --input-dir "$OUTPUT_DIR"
fi

echo "[run_experiment_causal] done"
echo "[run_experiment_causal] per-question JSONL: $OUTPUT_DIR/per_question"
echo "[run_experiment_causal] summary CSV: $OUTPUT_DIR/summary_by_condition.csv"
echo "[run_experiment_causal] summary by category CSV: $OUTPUT_DIR/summary_by_condition_category.csv"
echo "[run_experiment_causal] label-aware summary CSV: $OUTPUT_DIR/summary_label_aware_by_condition.csv"
echo "[run_experiment_causal] per-question CSV: $OUTPUT_DIR/per_question_metrics.csv"
echo "[run_experiment_causal] note: CSV outputs include BertScore columns"
