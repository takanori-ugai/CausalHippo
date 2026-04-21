# Optimizers

Top-level accessors in package `ragas`:

- `geneticOptimizer()`
- `dspyOptimizer(cache: CacheBackend? = null)`

## Core types

- `Optimizer`
- `OptimizationDataset`
- `OptimizationExample`
- `OptimizerPrompt` (`Text`, `MultiModal`)
- `OptimizerOutcome`

## Implementations

- `GeneticOptimizer`
- `DspyOptimizer` (adapter-backed with heuristic fallback)

Both implementations expose prompt-object optimization through `optimizePrompts(...)`.
