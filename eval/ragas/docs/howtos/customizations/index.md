<!-- Adapted for ragas-kotlin on 2026-04-01 -->
> [!NOTE]
> This page was adapted from `../docs/howtos/customizations/index.md` for the Kotlin port (`ragas-kotlin`).
> Python APIs/examples may not map 1:1. Use Kotlin entrypoints in package `ragas` and check [`/home/ugai/ragas/kotlin/PARITY_MATRIX.md`](/home/ugai/ragas/kotlin/PARITY_MATRIX.md) and [`/home/ugai/ragas/kotlin/MIGRATION.md`](/home/ugai/ragas/kotlin/MIGRATION.md).

# Customizations

How to customize various aspects of Ragas to suit your needs.

## General

- [Customize models](customize_models.md)
- [Customize timeouts, retries and others](./run_config.md)
- [Cancelling long-running tasks](cancellation.md)

## Metrics
- [Modify prompts in metrics](./metrics/_modifying-prompts-metrics.md)
- [Adapt metrics to target language](./metrics/metrics_language_adaptation.md)
- [Trace evaluations with Observability tools](metrics/tracing.md)


## Testset Generation
- [Generate test data from non-English corpus](testgenerator/_language_adaptation.md)
- [Configure or automatically generate Personas](testgenerator/_persona_generator.md)
- [Customize single-hop queries for RAG evaluation](testgenerator/_testgen-custom-single-hop.md)
- [Create custom multi-hop queries for RAG evaluation](testgenerator/_testgen-customisation.md)
- [Seed generations using production data](testgenerator/index.md)
