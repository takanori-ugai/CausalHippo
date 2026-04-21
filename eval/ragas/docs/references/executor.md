# Executor

`ragas.runtime.Executor` is the coroutine-based job runner used by evaluation.

## Highlights

- Concurrent execution bounded by `RunConfig.maxWorkers`
- Optional batching via `batchSize`
- Timeout and retry integration
- Optional fail-fast behavior via `raiseExceptions`
- Cooperative cancellation (`cancel()` / `isCancelled()`)

Most users will use it indirectly through `evaluate(...)` / `aevaluate(...)`.
