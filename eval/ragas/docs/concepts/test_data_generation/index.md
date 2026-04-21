<!-- Adapted for ragas-kotlin on 2026-04-01 -->
> [!NOTE]
> This page was adapted from `../docs/concepts/test_data_generation/index.md` for the Kotlin port (`ragas-kotlin`).
> Python APIs/examples may not map 1:1. Use Kotlin entrypoints in package `ragas` and check [`/home/ugai/ragas/kotlin/PARITY_MATRIX.md`](/home/ugai/ragas/kotlin/PARITY_MATRIX.md) and [`/home/ugai/ragas/kotlin/MIGRATION.md`](/home/ugai/ragas/kotlin/MIGRATION.md).

# Testset Generation

Curating a high quality test dataset is crucial for evaluating the performance of your AI application.

## Characteristics of an Ideal Test Dataset

- Contains high quality data samples
- Covers wide variety of scenarios as observed in real world.
- Contains enough number of samples to derive statistically significant conclusions.
- Continually updated to prevent data drift

Curating such a dataset manually can be time-consuming and expensive. Ragas provides a set of tools to generate synthetic test datasets for evaluating your AI applications.

<div class="grid cards" markdown>

- :fontawesome-solid-database:[__RAG__ for evaluating retrieval augmented generation pipelines](rag.md)
- :fontawesome-solid-robot: [__Agents or Tool use__ for evaluating agent workflows](agents.md)
</div>