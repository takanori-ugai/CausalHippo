# SWO69 — jar rebuild required (networked machine)

The offline box cannot rebuild `build/libs/causalrag-0.0.1-all.jar`: the
project requires Kotlin Gradle plugin **2.4.10**, but the local Gradle cache
only has **2.2.0**, and the wrapper wants Gradle 9.6.1 (cache has 8.14.3).
No network downloads are allowed on this box.

## What is missing from the current jar

Two runners were edited in source after the jar was built:

**`shared.eval.MultiConditionExperimentKt` (HippoRAG / RAG runner)** —
`src/main/kotlin/shared/eval/MultiConditionExperiment.kt`:

1. **E2 hook** — `--e2-snapshot-root <root>`: when set, the `hipporag_graph`
   condition loads `<root>/<sampleId>/` (a HippoRAG snapshot dir containing
   `working_dir/` with `graph.json` + embeddings) via `loadGraph` instead of
   upserting the corpus. Without it, **E2 on HippoRAG is impossible** —
   `run_experiment_swo69.sh` probes the jar for this flag
   (`jar_has_hippo_e2`) and skips hipporag in E2 automatically.
2. **T1 snapshot hook** — persist a `kg_snapshot/` copy (working_dir layout)
   so `scripts/analyze_kg_quality.py` can measure KG quality per sample.
   Without it, hipporag is absent from `kg_quality_per_sample.csv`.

**`shared.eval.graph.MultiConditionGraphExperimentKt` (graph runner)** —
`src/main/kotlin/shared/eval/graph/MultiConditionGraphExperiment.kt`:

3. **T1 hook for the non-unified GraphRAG condition** — copy the pipeline's
   `output/{entities,relationships}.parquet` into
   `<sampleRoot>/kg_snapshot/` after indexing. Without it, the non-unified
   graphrag condition (the one T6 E0/E1 use) has no KG-quality rows.

The graph runner (`shared.eval.graph.MultiConditionGraphExperimentKt`:
graphrag/lightrag/pathrag/youturag) **already has** its T1 `kg_snapshot`
hooks, `--extraction-llm-model` (E1), and `--e2-snapshot-root` (E2, LightRAG
`knowledge-graph.json` + `chunks.json`) in the current jar — verified by
string forensics on `causalrag-0.0.1-all.jar`.

## Rebuild steps (on a machine with network)

```bash
git pull   # includes the MultiConditionExperiment.kt changes
./gradlew shadowJar        # produces build/libs/causalrag-0.0.1-all.jar
```

Verify the hooks landed:

```bash
# hook 1 (RAG runner E2):
unzip -p build/libs/causalrag-0.0.1-all.jar \
  shared/eval/MultiConditionExperiment.class | strings | grep -c e2-snapshot-root
# expect >= 1
# hook 3 (graph runner graphrag T1) — "kg_snapshot" string count should rise
# by 1 vs the current jar (currently 4):
unzip -p build/libs/causalrag-0.0.1-all.jar \
  shared/eval/graph/MultiConditionGraphExperiment.class | strings | grep -c kg_snapshot
```

Copy the jar back to the offline box, overwriting
`CausalHippo/build/libs/causalrag-0.0.1-all.jar`, then:

- E2 can run with `E2_SYSTEMS=lightrag,hipporag bash run_experiment_swo69.sh e2`
- T1 analysis will include hipporag rows.

## Optional (not required for SWO69)

- Make LightRAG/PathRAG honor `--llm-model`/`OLLAMA_MODEL` env over the
  config JSON `shared.modelName` (config currently wins — see the
  `swo69_ollama_<model>.json` per-model config workaround in
  `run_experiment_swo69.sh: svo_config()`).

## What works without a rebuild

- E0 for all 5 systems (graphrunner systems + hipporag upsert path).
- E1 (all systems; LightRAG isolated via `--extraction-llm-model`, others compound).
- E2 on **LightRAG** (jar has the loader; `scripts/perturb_kg.py` emits the
  matching `<root>/<sid>/knowledge-graph.json` + `chunks.json` layout).
- T1/T3/T5 analysis on graph-runner outputs.
