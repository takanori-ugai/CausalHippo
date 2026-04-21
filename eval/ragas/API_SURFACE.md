# ragas-kotlin Public API (v0)

This file defines the intended stable entrypoints under package `ragas`.

## Top-level API

- `ragas.VERSION`
- `ragas.evaluate(...)`
- `ragas.aevaluate(...)`
- `ragas.defaultMetrics()`
- `ragas.tier1Metrics()`
- `ragas.tier2Metrics()`
- `ragas.tier3Metrics()`
- `ragas.tier4Metrics()`
- `ragas.geneticOptimizer()`
- `ragas.dspyOptimizer(cache?)`
- `ragas.withCache(llm, cache)`
- `ragas.withCache(embedding, cache)`
- `ragas.backendRegistry()`

## Stable Supporting Contracts

- Prompt typing stack under `ragas.prompt`:
  - `BasePrompt`, `TypedPrompt`, `FewShotTypedPrompt`, `DynamicFewShotTypedPrompt`
  - `FewShotPydanticPrompt`, `DynamicFewShotPydanticPrompt`
  - `ImageTextTypedPrompt`, `PromptContentPart`
- Evaluation parity hooks under `ragas.evaluation`:
  - `EvaluationCallback`, `EvaluationEvent`
  - `TokenUsage`, `CostEstimate`
  - `TokenUsageParser`, `CostParser`
- Optimizer prompt-object contracts under `ragas.optimizers`:
  - `OptimizerPrompt`, `PromptObjectEvaluator`, `OptimizerOutcome`
- Metric prompt optimization contract under `ragas.metrics.primitives`:
  - `OptimizableMetricPrompt`
- Backend extension/inspection contracts under `ragas.backends`:
  - `BackendDiscoveryProvider`
  - `BackendInfo`
  - `BackendRegistry.getBackendInfo(...)`, `listBackendInfo()`, `listAllNames()`

## Stability Goal

These entrypoints are intended to remain stable while internals evolve.
When adding modules, prefer exposing user-facing access through these facades.

## Notes

- `evaluate/aevaluate` delegate to the internal evaluation engine.
- If metrics are omitted, default single-turn metrics are used.
- Integrations remain partial; optimizer facades expose usable genetic + DSPy-style paths.
