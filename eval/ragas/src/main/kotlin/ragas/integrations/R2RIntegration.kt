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
 * Input record schema for R2R integration adapters.
 *
 * @property input User prompt text.
 * @property output Model output text.
 * @property retrievedContexts Retrieved context strings.
 * @property referenceContexts Optional reference context strings.
 * @property reference Optional reference answer.
 * @property metadata Optional record metadata.
 */
data class R2RRecord(
    val input: String,
    val output: String,
    val retrievedContexts: List<String> = emptyList(),
    val referenceContexts: List<String> = emptyList(),
    val reference: String? = null,
    val metadata: Map<String, String> = emptyMap(),
)

/**
 * Helper functions for evaluating R2R records with ragas metrics.
 */
object R2RIntegration {
    /**
     * Converts integration records into an evaluation dataset.
     *
     * @param records Integration records to process.
     */
    fun toDataset(records: List<R2RRecord>): EvaluationDataset<SingleTurnSample> =
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
        records: List<R2RRecord>,
        metrics: List<Metric>? = null,
        llm: BaseRagasLlm? = null,
        embeddings: BaseRagasEmbedding? = null,
        runConfig: RunConfig = RunConfig(),
        raiseExceptions: Boolean = false,
        runName: String = "ragas-r2r-evaluation",
        tags: Map<String, String> = emptyMap(),
        metadata: Map<String, String> = emptyMap(),
        observers: List<TraceObserver> = emptyList(),
    ): EvaluationResult =
        traceEvaluation(
            framework = "r2r",
            runName = runName,
            tags = tags,
            metadata = metadata,
            observers = observers,
        ) {
            unsupportedIntegration("r2r")
        }

    /**
     * Converts evaluation scores into integration-friendly metric rows.
     *
     * @param result Evaluation result payload.
     */
    fun toMetricPayload(result: EvaluationResult): List<Map<String, Any?>> = result.scores
}
