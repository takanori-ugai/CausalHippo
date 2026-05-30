# Experiments Guide

This document provides instructions for reproducing the experiment results presented in the paper. It covers dataset preparation, running multi-condition RAG experiments, and analyzing the results.

## Prerequisites

### 1. System Requirements

- **JDK**: 21 or newer.
- **Python**: 3.12.
- **Hardware**: Tested on a machine with 32GB RAM and no GPU.

### 2. Environment Setup

Set your LLM provider and API keys. OpenAI is recommended for the best results, especially for GraphRAG.

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key_here
export LLM_MODEL=gpt-5.4-mini  # or gpt-4o
export EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Build

The system must be built before running experiments:

```bash
./gradlew build
```

## Data Acquisition

**Note: Data preparation can be skipped as processed data is already stored in the `./data/` directory.**

### MuSiQue
Run the download script to fetch the MuSiQue v1.0 dataset:

```bash
./MuSiQue/download.sh
```

This will extract the dataset files into the `MuSiQue/data/` directory.

### Causal QA
The Causal QA datasets are located in `eval/causal/*.csv`.

### Webis-CausalQA
If you have the Webis-CausalQA dataset, place the CSV files in `Webis/Webis-CausalQA-22-v-2.0/input/original_splits/`.

---

## 1) Dataset Preparation

We use preparation scripts to create balanced subsets (default size 300) and compute metadata (hop counts, lexical overlap) for evaluation.

### MuSiQue Preparation

```bash
python3 scripts/prepare_musique_experiment_data.py \
  --input MuSiQue/data/musique_ans_v1.0_dev.jsonl \
  --output-dir data/musique_experiment \
  --balanced-size 300 \
  --seed 42
```

Expected output: `data/musique_experiment/musique_dev_balanced_300.jsonl`

### Causal-Reasoning-QA Preparation

```bash
python3 scripts/prepare_causal_experiment_data.py \
  --input-glob 'eval/causal/*.csv' \
  --output-dir data/causal_experiment \
  --target-size 300 \
  --seed 42

# Ensure the default path for runners exists
cp data/causal_experiment/causal_qa_balanced.jsonl \
   data/causal_experiment/causal_qa_balanced_300.jsonl
```

Expected output: `data/causal_experiment/causal_qa_balanced_300.jsonl`

### Webis-CausalQA-22 Preparation

```bash
python3 scripts/prepare_webis_experiment_data.py \
  --input-glob 'Webis/Webis-CausalQA-22-v-2.0/input/original_splits/*_train_*.csv' \
  --output-dir data/webis_experiment \
  --balanced-size 300
```

Expected output: `data/webis_experiment/webis_train_balanced_300.jsonl`

---

## 2) Running Experiments

We provide four primary experiment runners. All runners execute multiple RAG conditions in parallel and aggregate results into CSV files. Note that `gpt-5.4-mini` is the default model used in the paper.

### A. MuSiQue Multi-Condition RAG (Table 2)

Evaluates standard and causal RAG approaches on the MuSiQue dataset.

```bash
./run_experiment.sh \
  --data data/musique_experiment/musique_dev_balanced_300.jsonl \
  --conditions all \
  --top-k 5

# Unified adapters path
./run_experiment.sh \
  --data data/musique_experiment/musique_dev_balanced_300.jsonl \
  --conditions all \
  --top-k 5 \
  --use-unified-api
```

### B. Causal-Reasoning-QA Multi-Condition RAG (Table 3, 4)

Evaluates approaches on the original Causal-Reasoning-QA dataset.

```bash
./run_experiment_causal.sh \
  --data data/causal_experiment/causal_qa_balanced_300.jsonl \
  --conditions all \
  --top-k 5

# Unified adapters path
./run_experiment_causal.sh \
  --data data/causal_experiment/causal_qa_balanced_300.jsonl \
  --conditions all \
  --top-k 5 \
  --use-unified-api
```

### C. Webis-CausalQA-22 Multi-Condition RAG (Table 1)

Evaluates approaches on the Webis-CausalQA-22 dataset.

```bash
./run_experiment.sh \
  --data data/webis_experiment/webis_train_balanced_300.jsonl \
  --conditions all \
  --top-k 5

# Unified adapters path
./run_experiment.sh \
  --data data/webis_experiment/webis_train_balanced_300.jsonl \
  --conditions all \
  --top-k 5 \
  --use-unified-api
```

### D. Graph-based RAG Comparison (MuSiQue & Causal)

To reproduce Graph-based RAG baselines (LightRAG, PathRAG) as shown in the tables:

```bash
# For MuSiQue
./run_experiment_graph.sh \
  --data data/musique_experiment/musique_dev_balanced_300.jsonl \
  --config config/common_rag.json \
  --conditions all \
  --top-k 5 \
  --use-unified-api

# For Causal-Reasoning-QA
./run_experiment_causal_graph.sh \
  --data data/causal_experiment/causal_qa_balanced_300.jsonl \
  --config config/common_rag.json \
  --conditions all \
  --top-k 5 \
  --use-unified-api
```

## System Mappings in Paper

For the results presented in the submitted paper, the following mappings between the system names in tables and the experiment `Condition ID` are used:

| Paper System Name | Condition ID |
| :--- | :--- |
| **CausalHippo** | `causalhippo_ablation_no_rerank` |
| **CausalHippo-Fixed** | `causalhippo_fixed` |
| **causalrag** | `causalrag_fixed` |
| **hipporag** | `hipporag_graph` |

---

## 1) Dataset Preparation
...
## Condition IDs Reference

| Condition ID | Description | Runner(s) |
| :--- | :--- | :--- |
| `causalrag_fixed` | CausalRAG with fixed causal extraction. (**causalrag** in paper) | RAG |
| `causalrag_adapt` | CausalRAG with adaptive causal extraction. | RAG |
| `hipporag_graph` | HippoRAG using Graph-based retrieval. (**hipporag** in paper) | RAG |
| `hipporag_dpr` | HippoRAG using DPR-based retrieval. | RAG |
| `causalhippo_fixed` | Hybrid CausalHippo (Fixed). (**CausalHippo-Fixed** in paper) | RAG |
| `causalhippo_adaptive` | Hybrid CausalHippo (Adaptive). | RAG |
| `causalhippo_ablation_no_two_pass` | CausalHippoRAG ablation (dynamic weighting + confidence switch, no Two-Pass Adaptation). | RAG |
| `causalhippo_ablation_no_confidence_switch` | CausalHippoRAG ablation (dynamic weighting + Two-Pass Adaptation, no Confidence-Based Switch). | RAG |
| `causalhippo_ablation_no_two_pass_no_confidence_switch` | CausalHippoRAG ablation (dynamic weighting only; no Two-Pass Adaptation, no Confidence-Based Switch). | RAG |
| `causalhippo_ablation_no_rerank` | CausalHippoRAG without causal reranking. (**CausalHippo** in paper) | RAG |
| `lightrag` | LightRAG implementation. | Graph |
| `pathrag` | PathRAG implementation. | Graph |
| `graphrag` | Microsoft GraphRAG implementation. | Graph |

---

## 3) Evaluation Metrics

Experiments output detailed results in `eval_results/<experiment_name>_<timestamp>/`.

### Core Metrics

- **Exact Match (EM)**: Normalized prediction matches any gold answer.
- **F1 Score**: Token-level F1 between prediction and best gold answer.
- **BertScore**: Semantic similarity using BERT embeddings (Precision, Recall, F1).

### Retrieval Metrics

- **Support Recall @K**: Recall of supporting passages in top-K.
- **Bridge Coverage @5**: 1.0 if all supporting passages are in top-5.
- **MRR @5**: Reciprocal rank of the first supporting passage.
- **nDCG @5**: Normalized Discounted Cumulative Gain.

### RAGAS Metrics

- **Faithfulness**: Measures if the answer is derived solely from the context.
- **Response Groundedness**: Measures how well the response is grounded in retrieved passages.

### Latency & Efficiency

- **Index Latency**: Time spent indexing/reindexing (ms).
- **Query Latency**: Time spent retrieving + generating (ms).
- **Total Latency**: sum of the above.

### Causal-Specific Metrics

Available in `summary_label_aware_by_condition.csv` and `summary_by_condition_category.csv`:
- **Unknown Precision/Recall/F1**: Performance on detecting unanswerable/negative causal relations.
- **Label-Aware Means**: Metrics split by `label_true=true` and `label_true=false`.

---

## 4) Sensitivity Analysis (Figure 4)

We conduct a sensitivity analysis on the hyper-parameters $\alpha$ (`gatingPathCoef`: causal node weighting) and $\beta$ (`gatingCoverageCoef`: causal relationship strength weighting) to understand their impact on the `CausalHippo` performance.

### Running Sensitivity Sweep

To reproduce the sweep over the parameter space:

```bash
./scripts/sensitivity_sweep.sh
```

This script iterates through multiple combinations of $\alpha$ and $\beta$, saving aggregated results to `eval_results/sensitivity_analysis/`.

### Summary of Results (Sample)

The following table summarizes the performance across different configurations (extracted from `logs/sensitivity_sweep_revised.log` on a 10-sample subset):

| $\alpha$ | $\beta$ | F1 Score | Exact Match | Support Recall @5 | Avg Causal Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.0 | 0.0 | **0.667** | **0.50** | 0.80 | 0.173 |
| 0.0 | 2.0 | 0.437 | 0.20 | 0.80 | 0.295 |
| 0.0 | 4.0 | 0.637 | 0.40 | 0.80 | 0.275 |
| 2.0 | 0.0 | 0.467 | 0.30 | 0.80 | 0.337 |
| 2.0 | 2.0 | 0.300 | 0.10 | 0.80 | 0.359 |
| 2.0 | 4.0 | 0.540 | 0.40 | 0.80 | 0.327 |
| 4.0 | 0.0 | 0.424 | 0.20 | 0.80 | 0.298 |
| 4.0 | 2.0 | 0.573 | 0.30 | 0.80 | 0.384 |
| 4.0 | 4.0 | 0.580 | 0.40 | 0.80 | 0.332 |

*Note: Results on the full dataset (300 samples) as reported in `SensitivityAnalysis.md` show that $\alpha=0.0, \beta=4.0$ is the optimal configuration for causal reasoning tasks.*

### 5) Quantifying Stability and Robustness

To evaluate how consistently the system performs across diverse scenarios, we define several meta-metrics:
- **Stability ($\sigma_{F1}$)**: Standard deviation of F1-scores across benchmarks (lower is better).
- **Floor ($Min_{F1}$)**: The minimum performance bound across benchmarks (higher is better).
- **Robustness ($CV_{F1}$)**: Coefficient of Variation of F1-scores across the 20 logical categories in *Causal-Reasoning-QA* (lower is better).
- **Rank ($Avg_{MRR}$)**: Average ordinal rank based on MRR@5 across benchmarks (lower is better).

#### Summary of Stability & Robustness

| System | Stability ($\sigma_{F1}$) ↓ | Floor ($Min_{F1}$) ↑ | Robustness ($CV_{F1}$) ↓ | Rank ($Avg_{MRR}$) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **CausalHippo** | **0.064** | 0.361 | 0.434 | 1.67 |
| **CausalHippo-Fixed** | 0.070 | **0.372** | 0.450 | **1.33** |
| **CausalHippo (no Ph3&4)** | 0.068 | 0.359 | **0.401** | 2.00 |
| **causalrag** | 0.075 | 0.350 | 0.488 | 3.00 |
| **hipporag** | 0.120 | 0.033 | 0.867 | 5.00 |

*Data derived from the ISWC submission manuscript (Section 6.1).*

---

## Notes

- **Parallelism**: Use `--parallelism <n>` to control concurrent sample processing (default is 5 for RAG, 2 for Graph).
- **Output**: Per-question JSONL records are saved in the `per_question/` subdirectory of the experiment output.
- **Aggregation**: `scripts/aggregate_experiment_results.py` is called automatically unless `--skip-aggregate` is passed.

## Unified vs Legacy Parity Check

After running both legacy and unified experiments on the same dataset/conditions, compare per-question outputs:

```bash
python3 scripts/compare_unified_parity.py \
  --legacy-dir eval_results/multicondition_YYYYMMDD_HHMMSS \
  --unified-dir eval_results/multicondition_YYYYMMDD_HHMMSS_unified \
  --conditions all \
  --output-json eval_results/parity_report.json
```

Optional thresholded failure mode:

```bash
python3 scripts/compare_unified_parity.py \
  --legacy-dir eval_results/multicondition_legacy \
  --unified-dir eval_results/multicondition_unified \
  --max-mean-abs-diff 0.01 \
  --max-max-abs-diff 0.10 \
  --max-missing-ratio 0.0
```
