# Transforms

Transform orchestration for testset graph processing lives in `ragas.testset.transforms`.

## Core abstractions

- `BaseGraphTransformation`
- `Extractor`
- `Splitter`

## Composition

- `SingleTransform`
- `Parallel`
- `SequenceTransforms`
- `applyTransforms(kg, transforms, runConfig)`

Execution is coroutine-based and concurrency-limited by `RunConfig.maxWorkers`.
