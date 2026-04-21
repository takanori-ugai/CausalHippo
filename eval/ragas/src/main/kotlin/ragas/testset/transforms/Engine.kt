package ragas.testset.transforms

import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import ragas.runtime.RunConfig
import ragas.testset.graph.KnowledgeGraph

/**
 * Execution-plan node used to compose graph transformations.
 */
sealed interface Transforms

/**
 * Wrapper for a single transformation step.
 *
 * @property transform Transformation to execute.
 */
data class SingleTransform(
    val transform: BaseGraphTransformation,
) : Transforms

/**
 * Wrapper for transformations that can run in parallel.
 *
 * @property transformations Transformations to run concurrently.
 */
data class Parallel(
    val transformations: List<BaseGraphTransformation>,
) : Transforms

/**
 * Wrapper for an ordered transformation sequence.
 *
 * @property transformations Ordered transform plan nodes.
 */
data class SequenceTransforms(
    val transformations: List<Transforms>,
) : Transforms

/**
 * Applies a transform plan to the graph with concurrency controls from run config.
 *
 * @param kg Graph to mutate.
 * @param transforms Transformation plan tree.
 * @param runConfig Concurrency settings used when executing plan tasks.
 */
suspend fun applyTransforms(
    kg: KnowledgeGraph,
    transforms: Transforms,
    runConfig: RunConfig = RunConfig(),
) {
    when (transforms) {
        is SingleTransform -> {
            executePlan(transforms.transform.generateExecutionPlan(kg), runConfig)
        }

        is Parallel -> {
            val plan = transforms.transformations.flatMap { transform -> transform.generateExecutionPlan(kg) }
            executePlan(plan, runConfig)
        }

        is SequenceTransforms -> {
            transforms.transformations.forEach { transform ->
                applyTransforms(kg, transform, runConfig)
            }
        }
    }
}

private suspend fun executePlan(
    plan: List<suspend () -> Unit>,
    runConfig: RunConfig,
) {
    if (plan.isEmpty()) {
        return
    }

    coroutineScope {
        val semaphore = Semaphore(runConfig.maxWorkers)
        plan
            .map { task ->
                async {
                    semaphore.withPermit {
                        task()
                    }
                }
            }.awaitAll()
    }
}
