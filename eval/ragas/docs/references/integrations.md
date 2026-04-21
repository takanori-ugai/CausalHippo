# Integrations

## Record adapters

- `LangChainIntegration`
- `LlamaIndexIntegration`

Each provides:

- `toDataset(records)`
- `evaluateRecords(...)`
- `toMetricPayload(result)`

## Tracing observers

Evaluation helper paths can emit lifecycle events to `TraceObserver` implementations via integration tracing support.
