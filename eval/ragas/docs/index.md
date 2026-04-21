# Ragas Kotlin Documentation

This docs tree is the Kotlin-focused version of the upstream docs.

## What is implemented

- Core evaluation API: `evaluate(...)`, `aevaluate(...)`
- Dataset and sample models for single-turn and multi-turn flows
- Default metrics (`answer_relevancy`, `context_precision`, `faithfulness`, `context_recall`)
- Tiered metric collections (`tier1Metrics()` to `tier4Metrics()`)
- LangChain4j LLM and embedding adapters
- Prompt subsystem (`SimplePrompt`, typed prompts, dynamic few-shot)
- Cache wrappers for LLMs and embeddings
- Optimizers (`geneticOptimizer()`, `dspyOptimizer()`)
- Testset scaffold APIs (graph, transforms, generator)
- Integration adapters for LangChain and LlamaIndex records

## Start here

- [Get Started](./getstarted/index.md)
- [API References](./references/index.md)
- [Migration Notes](/home/ugai/ragas/kotlin/MIGRATION.md)
- [Parity Matrix](/home/ugai/ragas/kotlin/PARITY_MATRIX.md)
