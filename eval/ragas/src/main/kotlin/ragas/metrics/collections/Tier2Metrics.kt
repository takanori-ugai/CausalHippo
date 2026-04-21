package ragas.metrics.collections

import ragas.metrics.Metric

/**
 * Returns the Tier-2 agent/tool-call metric preset.
 *
 * This bundle focuses on agent execution quality: tool-call correctness,
 * goal achievement (with and without references), and workflow completion.
 */
fun agentToolCallTier2Metrics(): List<Metric> =
    listOf(
        ToolCallAccuracyMetric(),
        ToolCallF1Metric(),
        AgentGoalAccuracyWithReferenceMetric(),
        AgentGoalAccuracyWithoutReferenceMetric(),
        AgentWorkflowCompletionMetric(),
    )
