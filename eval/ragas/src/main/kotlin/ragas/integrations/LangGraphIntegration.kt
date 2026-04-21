package ragas.integrations

import ragas.embeddings.BaseRagasEmbedding
import ragas.integrations.tracing.TraceObserver
import ragas.llms.BaseRagasLlm
import ragas.metrics.Metric
import ragas.model.EvaluationDataset
import ragas.model.EvaluationResult
import ragas.model.SingleTurnSample
import ragas.runtime.RunConfig

/**
 * Input record schema for LangGraph integration adapters.
 *
 * @property input User prompt text.
 * @property output Model output text.
 * @property retrievedContexts Retrieved context strings.
 * @property referenceContexts Optional reference context strings.
 * @property reference Optional reference answer.
 * @property metadata Optional record metadata.
 */
data class LangGraphRecord(
    val input: String,
    val output: String,
    val retrievedContexts: List<String> = emptyList(),
    val referenceContexts: List<String> = emptyList(),
    val reference: String? = null,
    val metadata: Map<String, String> = emptyMap(),
)

/**
 * Helper functions for evaluating LangGraph records with ragas metrics.
 */
object LangGraphIntegration {
    /**
     * Converts integration records into an evaluation dataset.
     *
     * @param records Integration records to process.
     */
    fun toDataset(records: List<LangGraphRecord>): EvaluationDataset<SingleTurnSample> =
        EvaluationDataset(
            records.map { record ->
                SingleTurnSample(
                    userInput = record.input,
                    response = record.output,
                    retrievedContexts = record.retrievedContexts,
                    referenceContexts = record.referenceContexts.ifEmpty { null },
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
    @Suppress("UNUSED_PARAMETER")
    fun evaluateRecords(
        records: List<LangGraphRecord>,
        metrics: List<Metric>? = null,
        llm: BaseRagasLlm? = null,
        embeddings: BaseRagasEmbedding? = null,
        runConfig: RunConfig = RunConfig(),
        raiseExceptions: Boolean = false,
        runName: String = "ragas-langgraph-evaluation",
        tags: Map<String, String> = emptyMap(),
        metadata: Map<String, String> = emptyMap(),
        observers: List<TraceObserver> = emptyList(),
    ): EvaluationResult =
        traceEvaluation(
            framework = "langgraph",
            runName = runName,
            tags = tags,
            metadata = metadata,
            observers = observers,
        ) {
            unsupportedIntegration("langgraph")
        }

    /**
     * Converts evaluation scores into integration-friendly metric rows.
     *
     * @param result Evaluation result payload.
     */
    fun toMetricPayload(result: EvaluationResult): List<Map<String, Any?>> = result.scores
}
