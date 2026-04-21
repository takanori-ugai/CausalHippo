package ragas.integrations

import ragas.embeddings.BaseRagasEmbedding
import ragas.evaluate
import ragas.integrations.tracing.TraceObserver
import ragas.llms.BaseRagasLlm
import ragas.metrics.Metric
import ragas.model.EvaluationDataset
import ragas.model.EvaluationResult
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/**
 * Input record schema for LlamaIndex integration adapters.
 *
 * @property query User prompt text.
 * @property response Model output text.
 * @property retrievedNodes Retrieved node snippets.
 * @property referenceNodes Optional reference node snippets.
 * @property reference Optional reference answer.
 * @property metadata Optional record metadata.
 */
data class LlamaIndexRecord(
    val query: String,
    val response: String,
    val retrievedNodes: List<String> = emptyList(),
    val referenceNodes: List<String> = emptyList(),
    val reference: String? = null,
    val metadata: Map<String, String> = emptyMap(),
)

/**
 * Helper functions for evaluating LlamaIndex records with ragas metrics.
 */
object LlamaIndexIntegration {
    /**
     * Converts integration records into an evaluation dataset.
     *
     * @param records Integration records to process.
     */
    fun toDataset(records: List<LlamaIndexRecord>): EvaluationDataset<SingleTurnSample> =
        EvaluationDataset(
            records.map { record ->
                SingleTurnSample(
                    userInput = record.query,
                    response = record.response,
                    retrievedContexts = record.retrievedNodes,
                    referenceContexts = record.referenceNodes.ifEmpty { null },
                    reference = record.reference,
                )
            },
        )

    /**
     * Evaluates integration records with the selected metrics and model dependencies.
     *
     * @param records Integration records to process.
     * @param metrics Metrics to run.
     * @param llm LLM dependency used during generation/evaluation.
     * @param embeddings Embedding dependency used during evaluation.
     * @param runConfig Runtime retry/concurrency configuration.
     * @param raiseExceptions Whether metric failures should be thrown.
     * @param runName Logical run name used in tracing output.
     * @param tags Run-level tags.
     * @param metadata Run-level metadata.
     * @param observers Trace observers notified during execution.
     */
    fun evaluateRecords(
        records: List<LlamaIndexRecord>,
        metrics: List<Metric>? = null,
        llm: BaseRagasLlm? = null,
        embeddings: BaseRagasEmbedding? = null,
        runConfig: RunConfig = RunConfig(),
        raiseExceptions: Boolean = false,
        runName: String = "ragas-llamaindex-evaluation",
        tags: Map<String, String> = emptyMap(),
        metadata: Map<String, String> = emptyMap(),
        observers: List<TraceObserver> = emptyList(),
    ): EvaluationResult =
        traceEvaluation(
            framework = "llama_index",
            runName = runName,
            tags = tags,
            metadata = metadata,
            observers = observers,
        ) {
            evaluate(
                dataset = toDataset(records),
                metrics = metrics,
                llm = llm,
                embeddings = embeddings,
                runConfig = runConfig,
                raiseExceptions = raiseExceptions,
            )
        }

    /**
     * Converts evaluation scores into integration-friendly metric rows.
     *
     * @param result Evaluation result payload.
     */
    fun toMetricPayload(result: EvaluationResult): List<Map<String, Any?>> = result.scores
}
