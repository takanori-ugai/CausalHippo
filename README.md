# HybridRAG

This repository is primarily for `HybridRAG`: a two-stage retrieval-augmented generation pipeline that uses HippoRAG for broad candidate recall and a causal graph for reranking, explanation, and answer generation.

At a high level:

- HippoRAG handles graph-assisted semantic recall
- CausalRAG components build and query explicit causal pathways
- `HippoCausalRAGPipeline` combines both into the main HybridRAG flow

The project uses Kotlin/JVM, the Gradle wrapper, LangChain4j integrations, JTE prompt templates, KtLint, and Detekt.

## Requirements

- A working JDK
- Model credentials or local model endpoints for the provider you plan to use

For OpenAI-backed runs, `OPENAI_API_KEY` is the default fallback for both generation and embeddings.

## Build And Test

```bash
./gradlew build
./gradlew test
./gradlew ktlintCheck
./gradlew detekt
```

## HybridRAG CLI

The default application entry point is [`causalrag.examples.CliKt`](src/main/kotlin/causalrag/examples/Cli.kt), which now targets the HybridRAG pipeline.

Show the version:

```bash
./gradlew run --args="--version"
```

Index a text file or a directory of `.txt` files into a HybridRAG working directory:

```bash
./gradlew run --args="index --input data/docs --output tmp/hybrid-index --config config/causalrag.json"
```

Query a previously indexed HybridRAG directory:

```bash
./gradlew run --args="query --index tmp/hybrid-index --query \"What causes coastal flooding?\" --config config/causalrag.json"
```

Generate an interactive HTML visualization from a saved causal graph:

```bash
./gradlew run --args="visualize --index tmp/hybrid-index --output tmp/hybrid-index/graph.html"
```

Useful options for `index` and `query`:

- `--llm-model <model>`
- `--embedding-model <model>`
- `--semantic-mode <graph|dpr>`
- `--config <path>`

Useful options for `visualize`:

- `--index <dir>` or `--graph <path>`
- `--output <path>`
- `--highlight-nodes <a,b,c>`

The HybridRAG index directory stores at least:

- `graph.json` for the causal graph
- HippoRAG artifacts under the same output directory

## Configuration

The sample config at `config/causalrag.json` covers the shared pipeline settings used by the HybridRAG path:

- `modelName`
- `embeddingModel`
- `llmProvider`
- `llmApiKey`
- `llmBaseUrl`
- `embeddingApiKey`
- `templateStyle`

HybridRAG passes the effective model settings into HippoRAG as well, so one config can drive both stages.

### Prompt Style Behaviour

`templateStyle` controls which prompt template suffix is requested during answer generation.

- Default: `detailed`
- Shipped JTE styles with dedicated templates: `detailed`, `structured`, `chain_of_thought`
- If a style-specific JTE template is not present, the renderer falls back to the default JTE template: `causal_prompt.jte`
- In the current default setup, `basic` therefore uses the default JTE prompt, not a dedicated `basic` JTE template
- The older plain-text templates and hardcoded prompt builders are used as a fallback if JTE templates are not available.

## Main Pipeline

The primary orchestration class is [`HippoCausalRAGPipeline`](src/main/kotlin/causalrag/HippoCausalRAGPipeline.kt).

Its flow is:

1. Index documents into HippoRAG and the causal graph.
2. Retrieve candidates with HippoRAG semantic recall.
3. Combine semantic, causal, and BM25 signals.
4. Rerank passages with causal-path reasoning.
5. Build the final prompt and generate the answer.

## Examples

Run the direct HybridRAG example:

```bash
./gradlew execute -PmainClass=causalrag.examples.HippoCausalUsageKt
```

Run the older standalone causal pipeline example:

```bash
./gradlew execute -PmainClass=causalrag.examples.BasicUsageKt
```

Run the MusiQue batch evaluator:

```bash
./gradlew execute -PmainClass=causalrag.examples.MusiQue --args=""
```

The MusiQue evaluator expects `data/musique_ans_v1.0_train-200.jsonl` and reads settings from environment variables such as `OPENAI_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`, `EMBEDDING_MODEL`, `MUSIQUE_LIMIT`, and `MUSIQUE_PARALLELISM`.

Run the six-condition comparison experiment (CausalRAG / HippoRAG / CausalHippoRAG variants):

```bash
./run_experiment.sh --data data/musique_experiment/musique_dev_balanced_300.jsonl --conditions all --top-k 5
```

This writes per-question JSONL under `eval_results/multicondition_*/per_question/` and aggregated CSV files:
- `summary_by_condition.csv`
- `per_question_metrics.csv`
## OpenAlex Dataset Builder

Build an evaluation dataset similar to the CausalRAG paper setup using OpenAlex plus `gpt-5.4-mini` question generation:

```bash
export OPENAI_API_KEY=...
export OPENALEX_API_KEY=...
./gradlew execute -PmainClass=data.OpenAlexDatasetBuilderKt --args="--output data/openalex_eval_dataset.jsonl --max-docs 20 --questions-per-doc 5 --model gpt-5.4-mini --openalex-email you@example.com"
```

What it does:

- Pulls English article/proceedings records from OpenAlex with abstracts
- Reconstructs text from `abstract_inverted_index`
- Applies field-balanced sampling across detected top-level fields
- Generates grounded per-document questions with `gpt-5.4-mini`
- Writes JSONL records with metadata, text, and questions

Optional flags:

- `--per-page <1..200>` OpenAlex page size (default `50`)
- `--max-pages <n>` cap on OpenAlex pagination (default `20`)
- `--openalex-api-key <key>` OpenAlex API key (or set `OPENALEX_API_KEY`)
- `--provider <openai|ollama>` LLM provider (default `openai`)
- `--api-key <key>` provider API key override (otherwise uses `OPENAI_API_KEY`)

## HippoRAG Runner

The repository still includes the direct HippoRAG JSON runner at [`hipporag.MainKt`](src/main/kotlin/hipporag/Main.kt). Use it if you want to exercise HippoRAG by itself rather than the combined HybridRAG CLI.

```bash
./gradlew execute -PmainClass=hipporag.MainKt --args="--docs data/docs.json --queries data/queries.json --config config/hipporag.json"
```

Minimal input files:

`docs.json`

```json
[
  "Climate change causes rising sea levels.",
  "Rising sea levels increase coastal flooding."
]
```

`queries.json`

```json
[
  "What causes coastal flooding?"
]
```

## Repository Layout

```text
config/                  Sample configuration files
src/main/kotlin/causalrag   HybridRAG and causal graph code
src/main/kotlin/hipporag    HippoRAG implementation
src/test/kotlin/            Unit and smoke tests
```
