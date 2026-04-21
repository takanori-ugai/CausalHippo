package ragas.optimizers

import io.github.oshai.kotlinlogging.KotlinLogging
import ragas.prompt.PromptContentPart
import java.util.ServiceConfigurationError
import java.util.ServiceLoader

/**
 * Context snapshot passed to DSPy adapters during prompt candidate generation.
 *
 * @property iteration Iteration index.
 * @property dataset Dataset used for evaluation.
 * @property currentBestPrompt Current best prompt.
 */
data class DspyCompileContext(
    val iteration: Int,
    val dataset: OptimizationDataset,
    val currentBestPrompt: OptimizerPrompt,
)

/**
 * Optional adapter seam for plugging in external DSPy-backed candidate compilation.
 *
 * Implementations can be provided via Java ServiceLoader as an optional runtime dependency.
 */
fun interface DspyAdapter {
    /**
     * Proposes candidate prompt variants for the current optimization step.
     *
     * @param context Context object for candidate proposal.
     */
    fun proposeCandidates(context: DspyCompileContext): List<OptimizerPrompt>
}

/**
 * Service-loader utility that discovers and returns DSPy adapter implementations.
 */
object DspyAdapterLoader {
    private val logger = KotlinLogging.logger {}

    /**
     * Loads the first available DSPy adapter via service discovery, or null.
     */
    fun loadFirstOrNull(): DspyAdapter? =
        try {
            ServiceLoader.load(DspyAdapter::class.java).firstOrNull()
        } catch (error: ServiceConfigurationError) {
            logger.warn(error) {
                "DSPy adapter discovery failed. Proceeding without external DSPy adapter."
            }
            null
        }
}

internal class HeuristicDspyAdapter : DspyAdapter {
    /**
     * Proposes candidate prompt variants for the current optimization step.
     */
    override fun proposeCandidates(context: DspyCompileContext): List<OptimizerPrompt> =
        when (val prompt = context.currentBestPrompt) {
            is OptimizerPrompt.Text -> proposeTextCandidates(prompt)
            is OptimizerPrompt.MultiModal -> proposeMultimodalCandidates(prompt)
        }

    private fun proposeTextCandidates(prompt: OptimizerPrompt.Text): List<OptimizerPrompt.Text> {
        val base = prompt.value.trim()
        if (base.isEmpty()) return listOf(prompt)
        return listOf(
            OptimizerPrompt.Text("$base\nThink step-by-step, then answer with JSON only."),
            OptimizerPrompt.Text("$base\nUse retrieved context strictly and avoid unsupported claims."),
            OptimizerPrompt.Text("$base\nKeep the answer concise and deterministic."),
        )
    }

    private fun proposeMultimodalCandidates(prompt: OptimizerPrompt.MultiModal): List<OptimizerPrompt.MultiModal> {
        val base = prompt.content
        return listOf(
            OptimizerPrompt.MultiModal(
                base + PromptContentPart.Text("Think step-by-step, then answer with JSON only."),
            ),
            OptimizerPrompt.MultiModal(
                base + PromptContentPart.Text("Use retrieved context strictly and avoid unsupported claims."),
            ),
            OptimizerPrompt.MultiModal(
                base + PromptContentPart.Text("Keep the answer concise and deterministic."),
            ),
        )
    }
}
