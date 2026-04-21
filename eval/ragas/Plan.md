# RAGAS Kotlin Full-Parity Completion Plan

Last updated: 2026-04-02

## Goal

Complete Kotlin parity with Python `../src/ragas` so Kotlin can be used as a first-class equivalent for evaluation, prompts, metrics, integrations, backends, optimizers, and CLI workflows.

## Definition of Done

- Public Kotlin APIs cover Python-equivalent core workflows with compatible behavior.
- All Python default and collection metrics have Kotlin equivalents (or documented intentional exclusions).
- Prompt/optimizer/testset flows support Python-style typed prompt pipelines.
- Integration and backend surfaces are functionally mirrored at parity level.
- `PARITY_MATRIX.md` shows only intentional deferrals (no unowned “partial” gaps).

## Current Baseline (Already Completed)

- [x] Core Kotlin package, runtime, dataset/sample models, evaluator orchestration.
- [x] MVP default metrics (answer relevancy, context precision, faithfulness, context recall).
- [x] LangChain4j LLM/embedding adapters + cache wrappers.
- [x] Backends: in-memory, CSV, JSONL.
- [x] Prompt basics: `SimplePrompt`, dynamic few-shot selection, persistence, translation adapt().
- [x] Integration adapters: LangChain/LlamaIndex record-based evaluation.
- [x] Tracing scaffold: run lifecycle + in-memory/Langfuse-style/MLflow-style observers.
- [x] Testset scaffold: graph + transforms + generator baseline.
- [x] Test suite green (`./gradlew test`).

## Remaining Workstreams

### WS1: Evaluation API/Behavior Parity `[x]`

- [x] Extend `evaluate/aevaluate` parity options:
  - callbacks/tracing hook compatibility at API level
  - column remap path parity
  - token usage / cost callback parity hooks
  - executor-return / cancellable path parity where applicable
- [x] Add compatibility shims for Python-style argument patterns where safe.
- Progress note (2026-04-01):
  - Extended evaluator hook surface in `src/main/kotlin/ragas/evaluation/Evaluation.kt` and
    `src/main/kotlin/ragas/PublicApi.kt`:
    - callback/tracing event channel via `EvaluationCallback` / `EvaluationEvent`
    - column remap support (`columnMap`) with alias normalization (`question`/`query` -> `user_input`, etc.)
    - token usage callback path (`tokenUsageParser`) and cost callback path (`costParser`)
    - executor exposure/cancellation shim via `executorObserver`
  - Added tracking/cost support contracts in
    `src/main/kotlin/ragas/evaluation/EvaluationHooks.kt`:
    - `TokenUsage`, `CostEstimate`, parser type aliases
    - `TrackingRagasLlm` wrapper for parser-driven token accounting
  - Added focused WS1 parity coverage in
    `src/test/kotlin/ragas/EvaluationParityHooksTest.kt`:
    - `evaluateSupportsColumnRemapAliases`
    - `evaluateEmitsCallbacksAndTokenCostHooks`
    - `executorObserverCanCancelEvaluation`
- Exit criteria:
  - Kotlin evaluator supports Python-equivalent control surface for non-framework-specific options.

### WS2: Prompt System Parity `[ ]`

- [x] Implement typed prompt abstractions equivalent to Python `BasePrompt` + `PydanticPrompt`.
- [x] Add structured output parsing/validation loop (retry-on-parse-failure semantics).
- [x] Port few-shot typed prompt variants (including few-shot Pydantic flows).
- [x] Evaluate and scope multimodal prompt parity (`multi_modal_prompt.py`) for Kotlin compatibility.
- Progress note (2026-03-31):
  - Added WS2 foundation in `src/main/kotlin/ragas/prompt/TypedPrompt.kt`:
    `BasePrompt`, typed prompt model (`TypedPromptModel` / `TypedPromptExample`), `TypedPrompt`,
    `StructuredOutputParser`, and parse-retry generation flow.
  - Added coverage in `src/test/kotlin/ragas/prompt/TypedPromptTest.kt`.
  - Added few-shot typed variants in `src/main/kotlin/ragas/prompt/TypedPrompt.kt`:
    `FewShotTypedPrompt`, `DynamicFewShotTypedPrompt`, `FewShotPydanticPrompt`,
    and `DynamicFewShotPydanticPrompt`.
  - Extended `src/test/kotlin/ragas/prompt/TypedPromptTest.kt` with dynamic few-shot
    selection and few-shot Pydantic flow tests.
  - Multimodal parity scope (from Python `../src/ragas/prompt/multi_modal_prompt.py`):
    - Gap identified: Kotlin `BaseRagasLlm` is text-only (`generateText`) and currently has
      no multimodal message abstraction comparable to Python `PromptValue.to_messages()`.
    - Gap identified: Kotlin prompt stack has no typed image/text content block model and no
      secure image-source normalization pipeline (data URI / URL / local file policy).
    - Kotlin MVP scope (implement first):
      1. Add multimodal content model in prompt layer:
         `PromptContentPart.Text`, `PromptContentPart.ImageDataUri`, `PromptContentPart.ImageUrl`.
      2. Add multimodal-capable LLM interface extension:
         `MultiModalRagasLlm.generateContent(parts, ...)` while keeping `generateText` backward compatible.
      3. Add `ImageTextTypedPrompt` + few-shot variant on top of `BasePrompt` typed stack.
      4. Support data-URI images and already-hosted HTTPS image URLs; keep parsing-retry behavior from WS2.
      5. Add conformance tests for mixed text+image formatting and parser retry behavior.
    - Deferred (explicit, non-blocking for MVP):
      1. URL download/proxy/validation pipeline (SSRF checks, max-size streaming, content sniffing).
      2. Optional local file loading policy and directory allow-list behavior.
      3. Full Python callback/tracing parity for multimodal prompt generation path.
    - Exit target for multimodal scope:
      Kotlin can format and execute typed text+image prompts against multimodal-capable backends,
      with deterministic JSON output parsing/retry behavior equivalent to text prompts.
  - Implemented WS2 multimodal MVP:
    - Added prompt content-part model in `src/main/kotlin/ragas/prompt/PromptContentPart.kt`
      (`Text`, `ImageDataUri`, `ImageUrl` with HTTPS/data-URI validation).
    - Added multimodal LLM extension in
      `src/main/kotlin/ragas/llms/MultiModalRagasLlm.kt`.
    - Extended `src/main/kotlin/ragas/llms/LangChain4jLlm.kt` to implement multimodal generation.
    - Added `ImageTextTypedPrompt` in
      `src/main/kotlin/ragas/prompt/ImageTextTypedPrompt.kt` with retry-on-parse-failure semantics
      and text-only fallback for non-multimodal LLMs.
    - Added tests in
      `src/test/kotlin/ragas/prompt/ImageTextTypedPromptTest.kt` for mixed content assembly,
      multimodal retry flow, and fallback behavior.
- Exit criteria:
  - Metrics/optimizers can operate on typed prompt objects rather than string-only heuristics.

### WS3: Metrics Catalog Parity `[ ]`

- [ ] Port additional metric primitives/validators to support all collection metrics.
- [ ] Port Python metric collection modules in priority tiers:
  1. Retrieval/groundedness family
  2. Agent/tool-call metrics
  3. Text-quality/translation metrics
  4. Domain/rubric-specific metrics
- [ ] Align score semantics and edge-case handling with Python golden fixtures.
- Progress note (2026-04-01):
  - Started WS3 Tier-1 retrieval/groundedness ports in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/ContextRelevanceMetric.kt`
    - `src/main/kotlin/ragas/metrics/collections/ResponseGroundednessMetric.kt`
    - `src/main/kotlin/ragas/metrics/collections/ContextPrecisionCollectionMetrics.kt`
    - `src/main/kotlin/ragas/metrics/collections/EntityAndIdRetrievalMetrics.kt`
    - `src/main/kotlin/ragas/metrics/collections/Tier1Metrics.kt`
  - Added fixture baseline derived from Python migration tests:
    - `tests/e2e/metrics_migration/test_context_relevance_migration.py`
    - `tests/e2e/metrics_migration/test_response_groundedness_migration.py`
    - `tests/e2e/metrics_migration/test_context_precision_migration.py`
    - `tests/e2e/metrics_migration/test_context_entity_recall_migration.py`
    - `src/ragas/metrics/_context_precision.py::IDBasedContextPrecision`
    - Kotlin fixture: `src/test/resources/fixtures/metrics/ws3_tier1_retrieval_groundedness_fixture.json`
    - Kotlin fixture: `src/test/resources/fixtures/metrics/ws3_tier1_context_precision_fixture.json`
    - Kotlin fixture: `src/test/resources/fixtures/metrics/ws3_tier1_entity_id_fixture.json`
  - Added fixture-band conformance coverage:
    - `src/test/kotlin/ragas/metrics/collections/Tier1RetrievalGroundednessFixtureTest.kt`
    - `src/test/kotlin/ragas/metrics/collections/ContextPrecisionCollectionFixtureTest.kt`
    - `src/test/kotlin/ragas/metrics/collections/EntityAndIdRetrievalFixtureTest.kt`
  - Started WS3 Tier-2 agent/tool-call ports in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/AgentToolCallMetrics.kt`
      - `ToolCallAccuracyMetric` (strict/relaxed-order compatible path)
      - `ToolCallF1Metric`
    - `src/main/kotlin/ragas/metrics/collections/AgentWorkflowMetrics.kt`
      - `AgentGoalAccuracyWithReferenceMetric`
      - `AgentGoalAccuracyWithoutReferenceMetric`
      - `AgentWorkflowCompletionMetric`
    - `src/main/kotlin/ragas/metrics/collections/Tier2Metrics.kt`
      - `agentToolCallTier2Metrics()` accessor
    - `src/main/kotlin/ragas/PublicApi.kt`
      - public `tier2Metrics()` accessor
  - Added Tier-2 fixture baseline and score-band conformance coverage:
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier2_agent_tool_call_fixture.json`
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier2_agent_workflow_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/AgentToolCallFixtureTest.kt`
    - Test: `src/test/kotlin/ragas/metrics/collections/AgentWorkflowFixtureTest.kt`
    - API coverage update: `src/test/kotlin/ragas/PublicApiTest.kt`
  - Started WS3 Tier-3 answer-quality ports in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt`
      - `AnswerAccuracyMetric`
      - `AnswerCorrectnessMetric` (weighted factuality + similarity heuristic path)
      - `answerQualityTier3Metrics()` accessor
    - API coverage update:
      - `src/main/kotlin/ragas/PublicApi.kt`
      - public `tier3Metrics()` accessor
  - Added Tier-3 fixture baseline and score-band conformance coverage:
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier3_answer_quality_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/AnswerQualityFixtureTest.kt`
    - API coverage update: `src/test/kotlin/ragas/PublicApiTest.kt`
  - Continued WS3 Tier-3 with factual/topic ports in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/FactualAndTopicMetrics.kt`
      - `FactualCorrectnessMetric` (precision/recall/F1 modes; decomposition-level knobs)
      - `TopicAdherenceMetric` (precision/recall/F1 modes)
    - Extended Tier-3 accessor:
      - `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt`
      - `answerQualityTier3Metrics()` now includes factual/topic metrics
  - Added Tier-3 factual/topic fixture baseline and score-band conformance coverage:
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier3_factual_topic_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/FactualAndTopicFixtureTest.kt`
    - API coverage update: `src/test/kotlin/ragas/PublicApiTest.kt`
  - Continued WS3 Tier-3 with noise/summary ports in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/NoiseAndSummaryMetrics.kt`
      - `NoiseSensitivityMetric` (relevant/irrelevant mode)
      - `SummaryScoreMetric` (optional length penalty + coefficient)
    - Extended Tier-3 accessor:
      - `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt`
      - `answerQualityTier3Metrics()` now includes noise/summary metrics
  - Added Tier-3 noise/summary fixture baseline and score-band conformance coverage:
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier3_noise_summary_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/NoiseAndSummaryFixtureTest.kt`
    - API coverage update: `src/test/kotlin/ragas/PublicApiTest.kt`
  - Continued WS3 Tier-3 with quoted/chrf ports in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/QuotedAndChrfMetrics.kt`
      - `QuotedSpansAlignmentMetric` (quote span extraction + source containment checks)
      - `ChrfScoreMetric` (character n-gram F-score)
    - Extended Tier-3 accessor:
      - `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt`
      - `answerQualityTier3Metrics()` now includes quoted/chrf metrics
  - Added Tier-3 quoted/chrf fixture baseline and score-band conformance coverage:
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier3_quoted_chrf_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/QuotedAndChrfFixtureTest.kt`
    - API coverage update: `src/test/kotlin/ragas/PublicApiTest.kt`
  - Continued WS3 Tier-3 with BLEU/ROUGE ports in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/BleuAndRougeMetrics.kt`
      - `BleuScoreMetric`
      - `RougeScoreMetric` (fmeasure/precision/recall modes, rouge1/rougeL support)
    - Extended Tier-3 accessor:
      - `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt`
      - `answerQualityTier3Metrics()` now includes BLEU/ROUGE metrics
  - Added Tier-3 BLEU/ROUGE fixture baseline and score-band conformance coverage:
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier3_bleu_rouge_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/BleuAndRougeFixtureTest.kt`
    - API coverage update: `src/test/kotlin/ragas/PublicApiTest.kt`
  - Continued WS3 Tier-3 with semantic-similarity port in Kotlin:
    - `src/main/kotlin/ragas/metrics/collections/SemanticSimilarityMetric.kt`
      - `SemanticSimilarityMetric` (cosine-style token-vector similarity with optional threshold)
    - Extended Tier-3 accessor:
      - `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt`
      - `answerQualityTier3Metrics()` now includes semantic similarity
  - Added Tier-3 semantic-similarity fixture baseline and conformance coverage:
    - Fixture: `src/test/resources/fixtures/metrics/ws3_tier3_semantic_similarity_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/SemanticSimilarityFixtureTest.kt`
    - API coverage update: `src/test/kotlin/ragas/PublicApiTest.kt`
- Exit criteria:
  - Kotlin has parity for Python metrics currently under `../src/ragas/metrics`.

### WS4: Integrations Parity `[ ]`

- [ ] Expand beyond current LangChain/LlamaIndex record adapters.
- [ ] Port high-value integration modules (Langsmith, Helicone, Opik, LangGraph, Swarm, AG-UI, R2R, Bedrock as applicable).
- [ ] Preserve optional-dependency strategy and explicit unsupported errors for missing runtime deps.
- Exit criteria:
  - Kotlin integration surface matches Python’s practical integration coverage for mainstream workflows.

### WS5: Backends Parity `[x]`

- [x] Implement registry extension/discovery model equivalent to Python entry-point backend discovery.
- [x] Port optional Google Drive backend behavior or document platform-specific replacement.
- [x] Align aliasing/info/inspection features in backend registry.
- Progress note (2026-04-02):
  - Implemented Kotlin backend discovery SPI + lazy extension loading with `ServiceLoader`:
    - `src/main/kotlin/ragas/backends/BackendRegistry.kt`
      - Added `BackendDiscoveryProvider` and `discoverBackends(force = false)` with lazy discovery.
      - Added registry metadata model `BackendInfo` and inspection APIs:
        `getBackendInfo`, `listBackendInfo`, `listAllNames`, alias grouping.
      - Added `registerAliases(..., overwrite = false)` for parity-level alias management.
      - Extended registration metadata (`backendClass`, `description`, `source`) to support
        inspection and provenance (`builtin`, `runtime`, plugin/provider-defined).
  - Added WS5 discovery conformance coverage:
    - `src/test/kotlin/ragas/BackendsTest.kt`
      - `backendRegistryDiscoversServiceLoaderProviders`
      - `backendRegistryProvidesInspectionMetadata`
      - `backendRegistryListAllNamesIncludesAliases`
    - `src/test/kotlin/ragas/backends/TestBackendDiscoveryProvider.kt`
    - `src/test/resources/META-INF/services/ragas.backends.BackendDiscoveryProvider`
  - Explicit Google Drive backend decision for Kotlin:
    - Core module will not ship a built-in Google Drive backend.
    - Kotlin parity path uses optional plugin discovery (`BackendDiscoveryProvider`) so Google Drive
      can live in an external module with independent dependency/auth lifecycle.
    - Built-in backend baseline remains `inmemory`, `local/csv`, `local/jsonl`; callers can also
      pass concrete backend instances directly to experiment APIs.
- Exit criteria:
  - Backend registration and optional backend story are parity-level compatible.

### WS6: Testset Pipeline Parity `[x]`

- [x] Port production-grade transforms:
  - extractors (`llm_based`, `embeddings`, `regex`)
  - splitters
  - relationship builders
- [x] Port synthesizer variants (single-hop/multi-hop strategies and prompts).
- [x] Ensure generated testsets align structurally and semantically with Python output expectations.
- Progress note (2026-04-01):
  - Started WS6 production transform modules in Kotlin:
    - `src/main/kotlin/ragas/testset/transforms/Extractors.kt`
      - `LlmBasedSummaryExtractor`
      - `EmbeddingsTopicExtractor`
      - `RegexEntityExtractor`
    - `src/main/kotlin/ragas/testset/transforms/Splitters.kt`
      - `SentenceChunkSplitter`
    - `src/main/kotlin/ragas/testset/transforms/RelationshipBuilders.kt`
      - `AdjacentChunkRelationshipBuilder`
      - `SharedKeywordRelationshipBuilder`
    - Extended `src/main/kotlin/ragas/testset/transforms/BaseGraphTransformation.kt`
      with `RelationshipBuilder` base abstraction and dedupe-on-apply behavior.
  - Extended `src/main/kotlin/ragas/testset/synthesizers/TestsetGenerator.kt`:
    - chunk-aware single-hop synthesis (strategy-specific `single_hop_*` variants)
    - relationship-driven multi-hop synthesis (`multi_hop_overlap` / `multi_hop_sequence`) using `next`/`semantic_overlap` edges
    - transformed-graph candidate selection scoped to newly added document nodes
  - Added fixture-backed WS6 conformance coverage:
    - Fixture: `src/test/resources/fixtures/testset/ws6_synthesized_output_fixture.json`
    - Test: `src/test/kotlin/ragas/testset/WS6ProductionParityTest.kt`
    - Coverage validates graph structure (chunk + relationship counts), synthesizer mix,
      and output-quality guardrails (non-empty fields, minimum response length,
      lexical overlap with retrieved contexts).
  - Hardened synthesizer strategy layer in `src/main/kotlin/ragas/testset/synthesizers/TestsetGenerator.kt`:
    - Added deterministic sampling controls via `SynthesisControls` (`seed`, rank-biased toggle, single-hop/multi-hop target counts).
    - Added richer candidate ranking for single-hop and multi-hop samples.
    - Replaced coarse stub naming with strategy-specific synthesizers:
      `single_hop_entity`, `single_hop_topic`, `single_hop_summary`, `single_hop_chunk`,
      `multi_hop_overlap`, `multi_hop_sequence`.
    - Added deterministic conformance regression in
      `src/test/kotlin/ragas/testset/WS6ProductionParityTest.kt`
      (`seededSamplingProducesDeterministicSynthesizerSelection`).
  - Continued WS6 hardening pass in `src/main/kotlin/ragas/testset/synthesizers/TestsetGenerator.kt`:
    - Added explicit sampling modes (`TOP_K`, `RANK_BIASED`, `TEMPERATURE`) and temperature control.
    - Added document-diversity controls for single-hop selection (`enforceDocumentDiversity`, `maxSingleHopPerDocument`).
    - Added prompt-plan strategy builders (`buildSingleHopPromptPlan`, `buildMultiHopPromptPlan`) to make strategy prompts explicit and reusable.
    - Extended ranking signals with lexical-richness/length-balance and topic-bridge/summary-coverage features.
  - Expanded WS6 deterministic coverage in
    `src/test/kotlin/ragas/testset/WS6ProductionParityTest.kt`:
    - `temperatureSamplingModeIsDeterministicForSameSeed`
    - `documentDiversityCapLimitsSingleHopSamplesPerDocument`
  - Expanded WS6 edge-case fixture coverage in
    `src/test/kotlin/ragas/testset/WS6EdgeCaseFixturesTest.kt` with new fixtures:
    - sparse-overlap graph stress: `src/test/resources/fixtures/testset/ws6_sparse_overlap_fixture.json`
    - long-document splitting stress: `src/test/resources/fixtures/testset/ws6_long_document_fixture.json`
    - relationship-density bounds stress: `src/test/resources/fixtures/testset/ws6_relationship_density_fixture.json`
    - Coverage asserts semantic-overlap sparsity/density bounds, long-doc chunk/edge expectations,
      and conformance of synthesized sample structure under each scenario.
  - Added cross-language WS6 golden calibration:
    - Fixture: `src/test/resources/fixtures/testset/ws6_cross_language_golden_fixture.json`
    - Test: `src/test/kotlin/ragas/testset/WS6CrossLanguageGoldenTest.kt`
    - Calibration checks against Python-reference conventions for:
      - sample shape (`user_input`, `reference`, `reference_contexts`, `persona_name`, `query_style`, `query_length`)
      - prompt style families (single-hop and multi-hop marker templates)
      - graph statistics bounds (chunk counts, child/next edges, semantic-overlap density)
    - Kotlin generator updates for parity alignment:
      - `query_style` emitted as Python-style enum names (`PERFECT_GRAMMAR`, `WEB_SEARCH_LIKE`, etc.)
      - `query_length` emitted as Python-style enum names (`SHORT`, `MEDIUM`, `LONG`)
      - `persona_name` populated for synthesized samples
- Exit criteria:
  - Kotlin testset generation can replace Python flow for real-world synthesis scenarios.

### WS7: Optimizer Parity `[x]`

- [x] Implement DSPy optimizer path (optional dependency model, adapter layer, caching behavior).
- [x] Expand genetic optimizer from scaffold to Python-equivalent prompt optimization lifecycle.
- [x] Integrate optimizer outputs with metric prompt objects in typed prompt stack.
- [x] Wire early integration points so optimizers can consume typed/multimodal prompt objects.
- Progress note (2026-04-01):
  - Added prompt-object optimizer contracts in `src/main/kotlin/ragas/optimizers/Optimizer.kt`:
    - `OptimizerPrompt.Text` and `OptimizerPrompt.MultiModal(List<PromptContentPart>)`
    - `PromptObjectEvaluator`
    - `OptimizerOutcome`
  - Updated `src/main/kotlin/ragas/optimizers/GeneticOptimizer.kt` to optimize prompt objects directly:
    - typed crossover/mutation for text prompts
    - multimodal crossover/mutation across `PromptContentPart` lists
    - compatibility metadata in optimizer outcome
  - Kept string-based optimizer API compatibility by adapting legacy `optimize(...)` calls
    through the new prompt-object path.
  - Updated `src/main/kotlin/ragas/optimizers/DspyOptimizer.kt` to adopt the prompt-object
    optimizer interface as the base for the implemented DSPy-style path.
  - Added coverage in `src/test/kotlin/ragas/OptimizersTest.kt` for:
    - typed prompt-object optimization
    - multimodal prompt-object optimization
  - Completed DSPy path in Kotlin:
    - Added optional adapter seam via `src/main/kotlin/ragas/optimizers/DspyAdapter.kt`
      with `ServiceLoader` discovery (`DspyAdapterLoader`) and heuristic fallback adapter.
    - Implemented `src/main/kotlin/ragas/optimizers/DspyOptimizer.kt` candidate-compile loop
      over prompt objects with cache-backed score memoization.
    - Added top-level public API facades in `src/main/kotlin/ragas/PublicApi.kt`:
      `geneticOptimizer()` and `dspyOptimizer(cache?)`.
  - Integrated optimizer output application to metric prompt objects:
    - Added `src/main/kotlin/ragas/metrics/primitives/OptimizableMetricPrompt.kt`
      (`OptimizableMetricPrompt`, outcome apply helper, optimize-and-apply helper).
    - Updated `NumericMetric`, `DiscreteMetric`, and `RankingMetric` to store and consume
      mutable `OptimizerPrompt` objects.
  - Added coverage updates:
    - `src/test/kotlin/ragas/OptimizersTest.kt` now validates DSPy optimization and cache behavior.
    - `src/test/kotlin/ragas/metrics/primitives/StructuredFallbackTest.kt` validates
      optimizer-to-metric prompt application path.
    - `src/test/kotlin/ragas/PublicApiTest.kt` validates optimizer facades.
- Exit criteria:
  - Both genetic and DSPy optimization workflows are usable in Kotlin with parity semantics.

### WS8: CLI Parity `[x]`

- [x] Expand Kotlin CLI beyond `status/backends`:
  - experiment execution
  - metrics reporting/aggregation
  - baseline comparison and gate outputs
- [x] Keep CLI UX scriptable while mirroring essential Python CLI workflows.
- Progress note (2026-04-02):
  - Extended CLI entrypoint in `src/main/kotlin/ragas/cli/Main.kt` with parity workflow commands:
    - `eval`: run evaluation from JSON/JSONL dataset rows and emit structured report JSON.
    - `report`: aggregate metric means/non-null counts from an evaluation report.
    - `compare`: baseline vs candidate metric delta report with explicit gate thresholds.
  - Added scriptable behavior suitable for CI:
    - command exit code `2` for failed compare gates.
    - JSON output payloads for eval/report/compare commands (stdout or `--output` path).
    - deterministic option parsing via `--key value` pairs.
  - Added WS8 CLI conformance tests:
    - `src/test/kotlin/ragas/cli/CliParityTest.kt`
      - `evalCommandProducesReportJson`
      - `reportCommandAggregatesScores`
      - `compareCommandReturnsNonZeroWhenGateFails`
- Exit criteria:
  - Kotlin CLI can run parity-level evaluation workflows from terminal.

### WS9: Parity Verification + Documentation `[x]`

- [x] Build parity test matrix mapping Python module -> Kotlin module/test.
- [x] Add cross-language golden fixtures where behavior must be numerically/structurally aligned.
- [x] Update `README.md`, `PARITY_MATRIX.md`, `MIGRATION.md`, and `API_SURFACE.md` per milestone.
- [x] Add release checklist for parity claims.
- Progress note (2026-04-02):
  - Added versioned parity release-checklist document:
    - `RELEASE_CHECKLIST.md`
  - Checklist includes required sections for parity-claim readiness:
    - test evidence links (commands + artifact paths + key suites)
    - deferred-scope review (explicit rationale/risk/owner/next review)
    - versioned doc-freeze gates (release candidate and tag gates with consistency checks)
  - WS9 usage rule:
    - Any Kotlin parity claim for a tagged release must attach a completed checklist instance
      (filled values + links) in the release PR/notes.
- Progress note (2026-04-02, continuation):
  - Added Python->Kotlin parity test matrix with test/fixture evidence links:
    - `PARITY_TEST_MATRIX.md`
  - Added WS9 cross-language golden coverage for `Partial` metrics:
    - Fixture: `src/test/resources/fixtures/metrics/ws9_cross_language_partial_metrics_fixture.json`
    - Test: `src/test/kotlin/ragas/metrics/collections/WS9CrossLanguagePartialGoldenTest.kt`
  - Cross-language fixture scope includes currently `Partial` metrics listed in WS9 map:
    - Tier-2 workflow: `agent_goal_accuracy_with_reference`, `agent_goal_accuracy_without_reference`,
      `agent_workflow_completion`
    - Tier-3 defaults/collections:
      `context_recall`,
      `answer_accuracy`, `answer_correctness`, `factual_correctness`, `topic_adherence`,
      `noise_sensitivity`, `summary_score`, `quoted_spans_alignment`,
      `chrf_score`, `bleu_score`, `rouge_score`, `semantic_similarity`
- Intentional deferrals (tracked; not accidental gaps):
  - Multimodal ingestion hardening: URL download/proxy validation (SSRF/size/content checks) and optional local file policy.
  - Full production-grade testset synthesis parity beyond current WS6 baseline (broader transform/synthesizer coverage and deeper semantic parity against Python internals).
  - Broader integrations beyond current LangChain/LlamaIndex record adapters and trace observers.
  - Bundled (in-core) Google Drive backend implementation; Kotlin strategy is optional plugin module via backend discovery SPI.
  - Exact Python DSPy internals parity (Kotlin keeps adapter seam + heuristic fallback).
  - Full Python CLI UX surface beyond current Kotlin scriptable parity set (`eval`, `report`, `compare`, `status`, `backends`).
- WS9 parity map (full WS3 metrics module coverage):
  - Status legend:
    - `Done`: Kotlin target is implemented and covered by fixture/conformance tests.
    - `Partial`: Kotlin target exists but implementation semantics are intentionally simplified vs Python.
    - `Planned`: Kotlin target module is not implemented yet.

| Tier | Python file | Kotlin target | Status |
| --- | --- | --- | --- |
| Tier-1 | `../src/ragas/metrics/collections/context_relevance/metric.py` | `src/main/kotlin/ragas/metrics/collections/ContextRelevanceMetric.kt` (`ContextRelevanceMetric`) | Done |
| Tier-1 | `../src/ragas/metrics/collections/response_groundedness/metric.py` | `src/main/kotlin/ragas/metrics/collections/ResponseGroundednessMetric.kt` (`ResponseGroundednessMetric`) | Done |
| Tier-1 | `../src/ragas/metrics/collections/context_precision/metric.py` | `src/main/kotlin/ragas/metrics/collections/ContextPrecisionCollectionMetrics.kt` (`ContextPrecisionWithReferenceMetric`, `ContextPrecisionWithoutReferenceMetric`, `ContextPrecisionCollectionMetric`, `ContextUtilizationMetric`) | Done |
| Tier-1 | `../src/ragas/metrics/_context_precision.py` (`IDBasedContextPrecision`) | `src/main/kotlin/ragas/metrics/collections/EntityAndIdRetrievalMetrics.kt` (`IdBasedContextPrecisionMetric`) | Done |
| Tier-1 | `../src/ragas/metrics/collections/context_entity_recall/metric.py` | `src/main/kotlin/ragas/metrics/collections/EntityAndIdRetrievalMetrics.kt` (`ContextEntityRecallMetric`) | Done |
| Tier-2 | `../src/ragas/metrics/collections/tool_call_accuracy/metric.py` | `src/main/kotlin/ragas/metrics/collections/AgentToolCallMetrics.kt` (`ToolCallAccuracyMetric`) | Done |
| Tier-2 | `../src/ragas/metrics/collections/tool_call_f1/metric.py` | `src/main/kotlin/ragas/metrics/collections/AgentToolCallMetrics.kt` (`ToolCallF1Metric`) | Done |
| Tier-2 | `../src/ragas/metrics/collections/agent_goal_accuracy/metric.py` (`AgentGoalAccuracyWithReference`) | `src/main/kotlin/ragas/metrics/collections/AgentWorkflowMetrics.kt` (`AgentGoalAccuracyWithReferenceMetric`) | Partial |
| Tier-2 | `../src/ragas/metrics/collections/agent_goal_accuracy/metric.py` (`AgentGoalAccuracyWithoutReference`) | `src/main/kotlin/ragas/metrics/collections/AgentWorkflowMetrics.kt` (`AgentGoalAccuracyWithoutReferenceMetric`) | Partial |
| Tier-2 | `../src/ragas/metrics/collections/agent_goal_accuracy/metric.py` (workflow inference/completion intent) | `src/main/kotlin/ragas/metrics/collections/AgentWorkflowMetrics.kt` (`AgentWorkflowCompletionMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/answer_relevancy/metric.py` | `src/main/kotlin/ragas/metrics/defaults/AnswerRelevancyMetric.kt` (`AnswerRelevancyMetric`) | Done |
| Tier-3 | `../src/ragas/metrics/collections/faithfulness/metric.py` | `src/main/kotlin/ragas/metrics/defaults/FaithfulnessMetric.kt` (`FaithfulnessMetric`) | Done |
| Tier-3 | `../src/ragas/metrics/collections/context_recall/metric.py` | `src/main/kotlin/ragas/metrics/defaults/ContextRecallMetric.kt` (`ContextRecallMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/answer_accuracy/metric.py` | `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt` (`AnswerAccuracyMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/answer_correctness/metric.py` | `src/main/kotlin/ragas/metrics/collections/AnswerQualityMetrics.kt` (`AnswerCorrectnessMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/factual_correctness/metric.py` | `src/main/kotlin/ragas/metrics/collections/FactualAndTopicMetrics.kt` (`FactualCorrectnessMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/noise_sensitivity/metric.py` | `src/main/kotlin/ragas/metrics/collections/NoiseAndSummaryMetrics.kt` (`NoiseSensitivityMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/topic_adherence/metric.py` | `src/main/kotlin/ragas/metrics/collections/FactualAndTopicMetrics.kt` (`TopicAdherenceMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/quoted_spans/metric.py` | `src/main/kotlin/ragas/metrics/collections/QuotedAndChrfMetrics.kt` (`QuotedSpansAlignmentMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/summary_score/metric.py` | `src/main/kotlin/ragas/metrics/collections/NoiseAndSummaryMetrics.kt` (`SummaryScoreMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/chrf_score/metric.py` | `src/main/kotlin/ragas/metrics/collections/QuotedAndChrfMetrics.kt` (`ChrfScoreMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/_bleu_score.py` | `src/main/kotlin/ragas/metrics/collections/BleuAndRougeMetrics.kt` (`BleuScoreMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/_rouge_score.py` | `src/main/kotlin/ragas/metrics/collections/BleuAndRougeMetrics.kt` (`RougeScoreMetric`) | Partial |
| Tier-3 | `../src/ragas/metrics/collections/_semantic_similarity.py` | `src/main/kotlin/ragas/metrics/collections/SemanticSimilarityMetric.kt` (`SemanticSimilarityMetric`) | Partial |
| Tier-4 | `../src/ragas/metrics/collections/domain_specific_rubrics/metric.py` | `src/main/kotlin/ragas/metrics/collections/DomainSpecificRubricsMetrics.kt` (`DomainSpecificRubricsMetric`, `RubricsScoreWithReferenceMetric`, `RubricsScoreWithoutReferenceMetric`) | Done |
| Tier-4 | `../src/ragas/metrics/collections/instance_specific_rubrics/metric.py` | `src/main/kotlin/ragas/metrics/collections/InstanceSpecificRubricsMetric.kt` (`InstanceSpecificRubricsMetric`) | Done |
| Tier-4 | `../src/ragas/metrics/collections/sql_semantic_equivalence/metric.py` | `src/main/kotlin/ragas/metrics/collections/SqlSemanticEquivalenceMetric.kt` (`SqlSemanticEquivalenceMetric`) | Done |
| Tier-4 | `../src/ragas/metrics/collections/datacompy_score/metric.py` | `src/main/kotlin/ragas/metrics/collections/DataCompyScoreMetric.kt` (`DataCompyScoreMetric`) | Done |
| Tier-4 | `../src/ragas/metrics/collections/multi_modal_relevance/metric.py` | `src/main/kotlin/ragas/metrics/collections/MultiModalRelevanceMetric.kt` (`MultiModalRelevanceMetric`) | Done |
| Tier-4 | `../src/ragas/metrics/collections/multi_modal_faithfulness/metric.py` | `src/main/kotlin/ragas/metrics/collections/MultiModalFaithfulnessMetric.kt` (`MultiModalFaithfulnessMetric`) | Done |

- Exit criteria:
  - Parity claims are backed by repeatable tests and explicit documentation.

## Execution Order (Critical Path)

1. WS2 Prompt System Parity
2. WS3 Metrics Catalog Parity
3. WS7 Optimizer Parity
4. WS1 Evaluation API/Behavior Parity
5. WS6 Testset Pipeline Parity
6. WS4 Integrations Parity
7. WS5 Backends Parity
8. WS8 CLI Parity
9. WS9 Verification + Docs

## Milestone Checkpoints

- M1: Typed prompt stack + evaluator API parity baseline.
- M2: Core metric families ported with golden parity tests.
- M3: Optimizer parity (DSPy + upgraded genetic).
- M4: Testset pipeline parity.
- M5: Integrations/backends/CLI parity closure.
- M6: Final parity audit and docs freeze.

## Risks and Mitigations

- Risk: Python metrics depend on Python-only ecosystem packages.
  - Mitigation: isolate optional modules behind adapters and capability flags.
- Risk: Behavioral drift in prompt parsing and structured output.
  - Mitigation: fixture-first parity tests before exposing as stable.
- Risk: Scope explosion from low-value integrations.
  - Mitigation: prioritize by usage and mark explicit deferrals with rationale.

## Immediate Next Actions

1. Start WS4 integration parity expansion by prioritizing high-value adapters beyond LangChain/LlamaIndex (Langsmith/Helicone/Opik/LangGraph/R2R) with explicit unsupported fallbacks and focused conformance tests.
2. Run WS9 release-checklist execution for the next version candidate (attach `PARITY_TEST_MATRIX.md`, golden-fixture test evidence, and deferred-scope sign-off in release PR).
