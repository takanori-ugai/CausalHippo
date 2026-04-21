# Testset Schema

## Types

- `TestsetSample(evalSample: Sample, synthesizerName: String)`
- `Testset(samples: List<TestsetSample>, runId: String = UUID...)`

## Helpers

- `toEvaluationDataset()` converts a homogeneous testset to `EvaluationDataset`.
- `toList()` converts to row maps.
- `fromList(...)` currently supports single-turn deserialization only.
