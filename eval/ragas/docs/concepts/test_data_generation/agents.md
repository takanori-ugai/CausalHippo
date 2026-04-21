<!-- Adapted for ragas-kotlin on 2026-04-01 -->
> [!NOTE]
> This page was adapted from `../docs/concepts/test_data_generation/agents.md` for the Kotlin port (`ragas-kotlin`).
> Python APIs/examples may not map 1:1. Use Kotlin entrypoints in package `ragas` and check [`/home/ugai/ragas/kotlin/PARITY_MATRIX.md`](/home/ugai/ragas/kotlin/PARITY_MATRIX.md) and [`/home/ugai/ragas/kotlin/MIGRATION.md`](/home/ugai/ragas/kotlin/MIGRATION.md).

# Testset Generation for Agents or Tool use cases

Evaluating agentic or tool use workflows can be challenging as it involves multiple steps and interactions. It can be especially hard to curate a test suite that covers all possible scenarios and edge cases. We are working on a set of tools to generate synthetic test data for evaluating agent workflows.
