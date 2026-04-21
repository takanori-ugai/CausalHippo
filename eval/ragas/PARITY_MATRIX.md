# RAGAS Python -> Kotlin Parity Matrix

Last updated: 2026-04-02

## Core Runtime

| Area | Status | Notes |
| --- | --- | --- |
| Dataset/sample schemas | Done | Single-turn + multi-turn models with validation |
| Evaluation (`evaluate`/`aevaluate`) | Done | Core async/sync orchestration plus parity hooks: callback events, column remap, token/cost parser hooks, and executor observer for cancellable path |
| Run config/retry/executor | Done | Timeout/retry/batch/cancel behavior |
| Default metrics (MVP 4) | Done | answer_relevancy, context_precision, faithfulness, context_recall |
| Tier-1/2/3/4 metric accessors | Done | Public `tier1Metrics()`, `tier2Metrics()`, `tier3Metrics()`, `tier4Metrics()` are wired in `ragas.PublicApi` |

## Models and Providers

| Area | Status | Notes |
| --- | --- | --- |
| LLM adapter | Done | LangChain4j `ChatModel` wrapper |
| Embedding adapter | Done | LangChain4j `EmbeddingModel` wrapper |
| LLM/embedding caching | Done | In-memory cache wrappers |

## Supporting Modules

| Area | Status | Notes |
| --- | --- | --- |
| Backends | Done | `inmemory/csv/jsonl` built-ins plus lazy `ServiceLoader` backend discovery (`BackendDiscoveryProvider`), alias grouping, and backend inspection metadata; Google Drive is an explicit optional plugin strategy (not bundled in core) |
| Prompt subsystem | Partial | `SimplePrompt` + typed prompt stack (`TypedPrompt`, few-shot typed variants, structured parse-retry) + multimodal typed flow (`ImageTextTypedPrompt`, `PromptContentPart`, `MultiModalRagasLlm`) are implemented; multimodal URL/local-file ingestion hardening remains deferred |
| Testset/graph/transforms | Partial | Scaffold + core models + basic engine |
| Integrations | Partial | LangChain/LlamaIndex record adapters plus trace lifecycle observers (in-memory/Langfuse-style/MLflow-style); broader Python integrations are missing |
| CLI | Done | Scriptable parity workflow commands implemented: `eval` (run evaluation), `report` (metric aggregation), `compare` (baseline deltas + gate exit codes), plus `status/backends` |
| Optimizers | Done | Genetic + DSPy-style optimizer flows, prompt-object contracts (`OptimizerPrompt`), primitive metric prompt integration (`OptimizableMetricPrompt`), cache-backed DSPy scoring |

## Testing Parity

| Area | Status | Notes |
| --- | --- | --- |
| Unit tests | Done | Core, metrics, cache, backends, prompt, testset, multi-turn |
| Golden fixtures | Done | Default metrics + aggregation + WS3 tier fixture suites + WS9 cross-language partial-metrics fixture (`ws9_cross_language_partial_metrics_fixture.json`) |
| Parity test matrix | Done | Python->Kotlin module-to-test evidence map in `PARITY_TEST_MATRIX.md` |
| E2E evaluation flow | Done | Mock LLM + embeddings injected via evaluate; `./gradlew test` passing |

## Intentional Deferrals

- Evaluation API: remaining Python-specific ecosystem integrations are not yet mirrored.
- Multimodal prompt ingestion hardening: URL download/proxy validation (SSRF/size/content checks) and optional local file policy.
- Additional WS6 hardening beyond the current shipped baseline (broader transform/synthesizer coverage and deeper semantic parity against Python internals).
- Broader integrations beyond current LangChain/LlamaIndex adapters and tracing observers.
- Bundled core Google Drive backend implementation (strategy is optional plugin via backend discovery SPI).
- Exact Python DSPy runtime parity; Kotlin currently uses adapter seam + heuristic fallback semantics.
- Full Python CLI UX parity breadth remains intentionally narrowed; Kotlin covers essential scriptable evaluation/report/compare workflows.
