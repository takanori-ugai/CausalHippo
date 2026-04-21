# Ragas Kotlin

Kotlin port of the Ragas evaluation toolkit.

This module provides:
- evaluation pipeline (`evaluate` / `aevaluate`)
- default single-turn metrics
- WS3 metric collection accessors:
  - `tier1Metrics()` retrieval/groundedness family
  - `tier2Metrics()` agent/tool-call + workflow family
  - `tier3Metrics()` answer-quality/translation family
  - `tier4Metrics()` rubrics/advanced + multimodal family
- prompt subsystem:
  - `SimplePrompt` + dynamic few-shot + persistence/adapt
  - typed prompt stack (`TypedPrompt`, few-shot typed variants, structured parse-retry loop)
  - multimodal typed prompts (`ImageTextTypedPrompt`) with `PromptContentPart` text/image parts
- optimizer prompt-object flows:
  - `OptimizerPrompt.Text` / `OptimizerPrompt.MultiModal`
  - metric primitive integration via `OptimizableMetricPrompt`
  - public optimizer facades: `geneticOptimizer()`, `dspyOptimizer(cache?)`
- LangChain4j LLM/embedding adapters
- backend extension/discovery via `ServiceLoader` (`BackendDiscoveryProvider`)
- backend registry inspection metadata (`listBackendInfo`, `getBackendInfo`, aliases)
- CLI scaffold for runtime status and backend listing

## Requirements

- JDK 17+
- Gradle wrapper (included)

## Build

```bash
./gradlew build
```

## Test

```bash
./gradlew test
```

## Run CLI

The CLI entrypoint is `ragas.cli.MainKt`.

### Via Gradle `run`

```bash
./gradlew run --args="help"
./gradlew run --args="status"
./gradlew run --args="backends"
./gradlew run --args="eval --input dataset.json --metrics default --output run.json"
./gradlew run --args="report --input run.json"
./gradlew run --args="compare --baseline baseline.json --candidate run.json --gate faithfulness=0.01"
```

### Via Gradle `execute`

```bash
./gradlew execute --args="status"
```

## CLI Commands

### `help`

Show usage information.

```bash
./gradlew run --args="help"
```

### `status`

Show current Kotlin conversion/runtime status.

```bash
./gradlew run --args="status"
```

Current output sections include:
- core evaluation availability
- default metrics availability
- testset scaffold availability
- integration maturity

### `backends`

List registered backends from `BACKEND_REGISTRY`.

```bash
./gradlew run --args="backends"
```

### `eval`

Run evaluation from JSON/JSONL input rows and emit a report JSON.

```bash
./gradlew run --args="eval --input dataset.json --metrics default,tier1 --output run.json"
```

Supported input shapes:
- JSON array of row objects
- JSON object containing `rows` or `samples` array
- JSONL with one row-object per line (`--format jsonl` or `.jsonl` extension)

### `report`

Aggregate metric means/counts from a report JSON.

```bash
./gradlew run --args="report --input run.json --output summary.json"
```

### `compare`

Compare candidate report vs baseline report and optionally apply gate thresholds.
Returns exit code `2` when a gate fails (CI-friendly).

```bash
./gradlew run --args="compare --baseline baseline.json --candidate run.json --gate faithfulness=0.01,answer_relevancy=0.00"
```

## Backend Extensions

Kotlin supports plugin-style backend discovery using Java/Kotlin `ServiceLoader`.

1. Implement `ragas.backends.BackendDiscoveryProvider`.
2. Register one or more backends in `registerBackends(registry)`.
3. Add a service entry file:
   `META-INF/services/ragas.backends.BackendDiscoveryProvider`.

The registry lazily discovers providers on first access (`create`, `contains`, `availableNames`, etc.).

### Google Drive Backend Strategy

`ragas-kotlin` does not ship a built-in Google Drive backend in core.

- Reason: Drive/Sheets auth and runtime dependency surface are platform-specific and high-churn.
- Parity strategy: keep core backend-neutral, support discovery via `BackendDiscoveryProvider`, and
  implement Google Drive as an optional plugin module.
- Recommended usage today: `LocalCsvBackend`/`LocalJsonlBackend` for built-ins, or pass a custom
  backend instance directly to experiment APIs.

## API Quick Example

```kotlin
import ragas.evaluate
import ragas.model.EvaluationDataset
import ragas.model.SingleTurnSample

val dataset = EvaluationDataset(
    listOf(
        SingleTurnSample(
            userInput = "What is Kotlin?",
            response = "Kotlin is a JVM language.",
            retrievedContexts = listOf("Kotlin is a statically typed language for JVM and Android."),
            reference = "Kotlin is a statically typed language for JVM and Android."
        )
    )
)

val result = evaluate(dataset = dataset)
println(result.scores)
```

## Notes

- This Kotlin module is under active parity work with the Python implementation in `../`.
- See [`Plan.md`](./Plan.md), [`PARITY_MATRIX.md`](./PARITY_MATRIX.md),
  [`PARITY_TEST_MATRIX.md`](./PARITY_TEST_MATRIX.md), and [`MIGRATION.md`](./MIGRATION.md)
  for detailed status and parity evidence mapping.
