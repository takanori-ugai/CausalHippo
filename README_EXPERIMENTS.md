# Experiments Guide

This document provides instructions for reproducing the experiment results presented in the paper. It covers dataset preparation, running multi-condition RAG experiments, and analyzing the results.

## Prerequisites

### 1. Environment Setup

Set your LLM provider and API keys. OpenAI is recommended for the best results, especially for GraphRAG.

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key_here
export LLM_MODEL=gpt-5.4-mini  # or gpt-4o
export EMBEDDING_MODEL=text-embedding-3-small
```

### 2. Data Acquisition

#### MuSiQue
Run the download script to fetch the MuSiQue v1.0 dataset:

```bash
./MuSiQue/download.sh
```

This will extract the dataset files into the `MuSiQue/data/` directory.

#### Causal QA
The Causal QA datasets are located in `eval/causal/*.csv`.

#### Webis-CausalQA
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
```

### B. Causal-Reasoning-QA Multi-Condition RAG (Table 3, 4)

Evaluates approaches on the original Causal-Reasoning-QA dataset.

```bash
./run_experiment_causal.sh \
  --data data/causal_experiment/causal_qa_balanced_300.jsonl \
  --conditions all \
  --top-k 5
```

### C. Webis-CausalQA-22 Multi-Condition RAG (Table 1)

Evaluates approaches on the Webis-CausalQA-22 dataset.

```bash
./run_experiment.sh \
  --data data/webis_experiment/webis_train_balanced_300.jsonl \
  --conditions all \
  --top-k 5
```

### D. Graph-based RAG Comparison (MuSiQue & Causal)

To reproduce Graph-based RAG baselines (LightRAG, PathRAG) as shown in the tables:

```bash
# For MuSiQue
./run_experiment_graph.sh \
  --data data/musique_experiment/musique_dev_balanced_300.jsonl \
  --conditions all \
  --top-k 5

# For Causal-Reasoning-QA
./run_experiment_causal_graph.sh \
  --data data/causal_experiment/causal_qa_balanced_300.jsonl \
  --conditions all \
  --top-k 5
```

---

## Condition IDs Reference

| Condition ID | Description | Runner(s) |
| :--- | :--- | :--- |
| `causalrag_fixed` | CausalRAG with fixed causal extraction. | RAG |
| `causalrag_adapt` | CausalRAG with adaptive causal extraction. | RAG |
| `hipporag_graph` | HippoRAG using Graph-based retrieval. | RAG |
| `hipporag_dpr` | HippoRAG using DPR-based retrieval. | RAG |
| `causalhippo_fixed` | Hybrid CausalHippo (Fixed). | RAG |
| `causalhippo_adaptive` | Hybrid CausalHippo (Adaptive). | RAG |
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

## Notes

- **Parallelism**: Use `--parallelism <n>` to control concurrent sample processing (default is 5 for RAG, 2 for Graph).
- **Output**: Per-question JSONL records are saved in the `per_question/` subdirectory of the experiment output.
- **Aggregation**: `scripts/aggregate_experiment_results.py` is called automatically unless `--skip-aggregate` is passed.
