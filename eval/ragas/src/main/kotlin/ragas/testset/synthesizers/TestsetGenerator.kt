package ragas.testset.synthesizers

import ragas.model.SingleTurnSample
import ragas.testset.graph.KnowledgeGraph
import ragas.testset.graph.Node
import ragas.testset.graph.NodeType
import ragas.testset.graph.Relationship
import ragas.testset.transforms.PropertyNames
import ragas.testset.transforms.SequenceTransforms
import ragas.testset.transforms.Transforms
import ragas.testset.transforms.applyTransforms
import kotlin.math.exp
import kotlin.random.Random

/**
 * Sampling strategy used when selecting synthesis candidates.
 */
enum class SamplingMode {
    /** Always take the highest-scoring candidates first. */
    TOP_K,

    /** Sample with probability biased toward higher-ranked candidates. */
    RANK_BIASED,

    /** Sample stochastically by softmax over scores and temperature. */
    TEMPERATURE,
}

/**
 * Controls stochastic and balancing behavior for testset synthesis.
 *
 * @property seed Random seed used for deterministic selection when desired.
 * @property samplingMode Explicit sampling mode override.
 * @property useRankBiasedSampling Whether to default to [SamplingMode.RANK_BIASED] when [samplingMode] is null.
 * @property temperature Softmax temperature used by temperature sampling.
 * @property enforceDocumentDiversity Whether to cap per-document single-hop samples.
 * @property maxSingleHopPerDocument Maximum single-hop samples per document when diversity is enabled.
 * @property singleHopCount Optional requested number of single-hop samples.
 * @property multiHopCount Optional requested number of multi-hop samples.
 */
data class SynthesisControls(
    val seed: Int = 42,
    val samplingMode: SamplingMode? = null,
    val useRankBiasedSampling: Boolean = true,
    val temperature: Double = 0.8,
    val enforceDocumentDiversity: Boolean = true,
    val maxSingleHopPerDocument: Int = 2,
    val singleHopCount: Int? = null,
    val multiHopCount: Int? = null,
)

/**
 * Builds a knowledge graph from documents and synthesizes evaluation samples from it.
 *
 * @property knowledgeGraph Mutable working graph used during generation.
 */
class TestsetGenerator(
    var knowledgeGraph: KnowledgeGraph = KnowledgeGraph(),
) {
    /**
     * Constructs graph nodes/edges from raw documents and generates a testset of the requested size.
     *
     * @param documents Source document texts.
     * @param testsetSize Maximum number of generated samples to return.
     * @param transforms Optional transform pipeline applied after document nodes are added.
     * @param synthesisControls Candidate scoring and sampling controls.
     * @return Generated [Testset].
     */
    suspend fun generateFromDocuments(
        documents: List<String>,
        testsetSize: Int,
        transforms: Transforms? = null,
        synthesisControls: SynthesisControls = SynthesisControls(),
    ): Testset {
        val nodes =
            documents.map { text ->
                Node(
                    type = NodeType.DOCUMENT,
                    properties = mutableMapOf("page_content" to text),
                )
            }

        val kg = knowledgeGraph
        kg.nodes += nodes
        if (transforms != null) {
            applyTransforms(kg, transforms)
        }
        knowledgeGraph = kg

        val random = Random(synthesisControls.seed)
        val documentIds = nodes.map { node -> node.id }.toSet()
        val generatedChunks =
            generatedChunksForDocuments(
                kg = kg,
                documentIds = documentIds,
            )
        val singleHopCandidates =
            if (generatedChunks.isNotEmpty()) {
                generatedChunks
            } else {
                nodes.filter { node -> !node.getProperty("page_content").isNullOrBlank() }
            }

        val singleHopScored =
            scoreSingleHopCandidates(
                candidates = singleHopCandidates,
                relationships = kg.relationships,
            )
        val multiHopScored =
            scoreMultiHopCandidates(
                chunks = generatedChunks,
                relationships = kg.relationships,
                singleHopScores = singleHopScored.associate { (candidate, score) -> candidate.id to score },
            )

        val defaultMultiHopTarget =
            if (generatedChunks.size >= 2 && testsetSize >= 3) {
                1
            } else {
                0
            }
        val requestedMultiHop = synthesisControls.multiHopCount ?: defaultMultiHopTarget
        val multiHopTarget = requestedMultiHop.coerceIn(0, testsetSize)
        val selectedMultiHop =
            selectRanked(
                scored = multiHopScored,
                count = multiHopTarget,
                random = random,
                controls = synthesisControls,
                groupKey = { candidate -> multiHopGroupKey(candidate) },
                maxPerGroup = 1,
            )
        val multiHopSamples = generateMultiHopSamples(selectedMultiHop)

        val remaining = (testsetSize - multiHopSamples.size).coerceAtLeast(0)
        val requestedSingleHop = synthesisControls.singleHopCount ?: remaining
        val singleHopTarget = requestedSingleHop.coerceIn(0, remaining)
        val selectedSingleHop =
            selectRanked(
                scored = singleHopScored,
                count = singleHopTarget,
                random = random,
                controls = synthesisControls,
                groupKey = { node -> singleHopGroupKey(node) },
                maxPerGroup =
                    if (synthesisControls.enforceDocumentDiversity) {
                        synthesisControls.maxSingleHopPerDocument.coerceAtLeast(1)
                    } else {
                        Int.MAX_VALUE
                    },
            )
        val singleHopSamples = generateSingleHopSamples(selectedSingleHop)
        val evalSamples = (multiHopSamples + singleHopSamples).take(testsetSize)

        return Testset(samples = evalSamples)
    }

    private fun generatedChunksForDocuments(
        kg: KnowledgeGraph,
        documentIds: Set<String>,
    ): List<Node> {
        val childRelationships =
            kg.relationships.filter { rel ->
                rel.type == "child" && rel.sourceId in documentIds
            }

        val chunks =
            childRelationships
                .mapNotNull { rel -> kg.getNodeById(rel.targetId) }
                .filter { node -> node.type == NodeType.CHUNK }
                .filter { node -> !node.getProperty("page_content").isNullOrBlank() }
                .distinctBy { node -> node.id }

        return chunks.sortedBy { chunk ->
            val parent = chunk.getProperty("parent_document_id").orEmpty()
            val index = chunk.getProperty("chunk_index")?.toIntOrNull() ?: Int.MAX_VALUE
            "$parent:$index"
        }
    }

    private fun scoreSingleHopCandidates(
        candidates: List<Node>,
        relationships: List<Relationship>,
    ): List<Pair<Node, Double>> {
        val degreeByNodeId =
            relationships
                .flatMap { rel -> listOf(rel.sourceId, rel.targetId) }
                .groupingBy { nodeId -> nodeId }
                .eachCount()

        return candidates
            .map { node ->
                val content = node.getProperty("page_content").orEmpty()
                val summary = nodeSummary(node)
                val entities = nodeEntities(node)
                val topic = nodeTopic(node)
                val tokenStats = tokenStats(content)
                val contentScore = (content.length.coerceAtMost(320) / 320.0)
                val summaryBonus = if (summary.isNotBlank()) 0.25 else 0.0
                val entityBonus = if (entities.isNotBlank()) 0.2 else 0.0
                val topicBonus = if (topic.isNotBlank()) 0.15 else 0.0
                val degreeBonus = ((degreeByNodeId[node.id] ?: 0).coerceAtMost(4)) * 0.05
                val lexicalRichnessBonus = tokenStats.uniqueRatio * 0.2
                val balancedLengthBonus = if (tokenStats.total in 12..80) 0.08 else 0.0
                node to (contentScore + summaryBonus + entityBonus + topicBonus + degreeBonus + lexicalRichnessBonus + balancedLengthBonus)
            }.sortedWith(compareByDescending<Pair<Node, Double>> { it.second }.thenBy { it.first.id })
    }

    private data class MultiHopCandidate(
        val left: Node,
        val right: Node,
        val relationship: Relationship,
    )

    private fun scoreMultiHopCandidates(
        chunks: List<Node>,
        relationships: List<Relationship>,
        singleHopScores: Map<String, Double>,
    ): List<Pair<MultiHopCandidate, Double>> {
        val chunksById = chunks.associateBy { node -> node.id }
        return relationships
            .asSequence()
            .filter { rel -> rel.type == "semantic_overlap" || rel.type == "next" }
            .mapNotNull { rel ->
                val left = chunksById[rel.sourceId] ?: return@mapNotNull null
                val right = chunksById[rel.targetId] ?: return@mapNotNull null
                val base = ((singleHopScores[left.id] ?: 0.0) + (singleHopScores[right.id] ?: 0.0)) / 2.0
                val leftTopic = nodeTopic(left)
                val rightTopic = nodeTopic(right)
                val topicBridgeBonus =
                    if (leftTopic.isNotBlank() && rightTopic.isNotBlank() && leftTopic != rightTopic) {
                        0.1
                    } else {
                        0.0
                    }
                val summaryCoverageBonus =
                    listOf(nodeSummary(left), nodeSummary(right))
                        .count { summary -> !summary.isNullOrBlank() } * 0.06
                val edgeBonus =
                    when (rel.type) {
                        "semantic_overlap" -> {
                            val shared = rel.properties["shared_keyword_count"]?.toIntOrNull() ?: 1
                            0.25 + shared * 0.08
                        }

                        "next" -> {
                            val sameParent =
                                left.getProperty("parent_document_id") == right.getProperty("parent_document_id")
                            if (sameParent) 0.22 else 0.12
                        }

                        else -> {
                            0.0
                        }
                    }
                MultiHopCandidate(left = left, right = right, relationship = rel) to
                    (base + edgeBonus + topicBridgeBonus + summaryCoverageBonus)
            }.sortedWith(compareByDescending<Pair<MultiHopCandidate, Double>> { it.second }.thenBy { it.first.left.id + it.first.right.id })
            .toList()
    }

    private fun <T> selectRanked(
        scored: List<Pair<T, Double>>,
        count: Int,
        random: Random,
        controls: SynthesisControls,
        groupKey: (T) -> String,
        maxPerGroup: Int,
    ): List<T> {
        if (count <= 0 || scored.isEmpty()) {
            return emptyList()
        }
        val mode = resolveSamplingMode(controls)
        val pool = if (count >= scored.size) scored else scored.toList()

        val selected =
            when (mode) {
                SamplingMode.TOP_K -> {
                    pool.take(count).map { (item) -> item }
                }

                SamplingMode.RANK_BIASED -> {
                    selectRankBiased(
                        scored = pool,
                        count = count,
                        random = random,
                    )
                }

                SamplingMode.TEMPERATURE -> {
                    selectTemperature(
                        scored = pool,
                        count = count,
                        random = random,
                        temperature = controls.temperature,
                    )
                }
            }

        return applyDiversityCap(
            selected = selected,
            backup = scored.map { (item) -> item },
            count = count,
            groupKey = groupKey,
            maxPerGroup = maxPerGroup,
        )
    }

    private fun resolveSamplingMode(controls: SynthesisControls): SamplingMode =
        controls.samplingMode
            ?: if (controls.useRankBiasedSampling) {
                SamplingMode.RANK_BIASED
            } else {
                SamplingMode.TOP_K
            }

    private fun <T> selectRankBiased(
        scored: List<Pair<T, Double>>,
        count: Int,
        random: Random,
    ): List<T> {
        if (count >= scored.size) {
            return scored.map { (item) ->
                item
            }
        }

        val pool = scored.toMutableList()
        val selected = mutableListOf<T>()
        repeat(count) {
            val chosenIndex = selectRankBiasedIndex(pool.size, random)
            selected += pool.removeAt(chosenIndex).first
            if (pool.isEmpty()) {
                return@repeat
            }
        }
        return selected
    }

    private fun <T> selectTemperature(
        scored: List<Pair<T, Double>>,
        count: Int,
        random: Random,
        temperature: Double,
    ): List<T> {
        if (count >= scored.size) {
            return scored.map { (item) -> item }
        }
        val normalizedTemperature = temperature.coerceAtLeast(0.05)
        val pool = scored.toMutableList()
        val selected = mutableListOf<T>()

        repeat(count) {
            if (pool.isEmpty()) {
                return@repeat
            }
            val maxScore = pool.maxOf { (_, score) -> score }
            val weights =
                pool.map { (_, score) ->
                    exp((score - maxScore) / normalizedTemperature)
                }
            val totalWeight = weights.sum().coerceAtLeast(1e-9)
            var ticket = random.nextDouble() * totalWeight
            var chosenIndex = 0
            for (index in weights.indices) {
                ticket -= weights[index]
                if (ticket <= 0.0) {
                    chosenIndex = index
                    break
                }
            }
            selected += pool.removeAt(chosenIndex).first
        }

        return selected
    }

    private fun <T> applyDiversityCap(
        selected: List<T>,
        backup: List<T>,
        count: Int,
        groupKey: (T) -> String,
        maxPerGroup: Int,
    ): List<T> {
        require(maxPerGroup > 0 || maxPerGroup == Int.MAX_VALUE) {
            "maxPerGroup must be positive or Int.MAX_VALUE, got $maxPerGroup"
        }
        if (maxPerGroup == Int.MAX_VALUE) {
            return selected.take(count)
        }
        val groupCounts = mutableMapOf<String, Int>()
        val result = mutableListOf<T>()

        fun tryAdd(item: T): Boolean {
            val key = groupKey(item)
            val used = groupCounts[key] ?: 0
            if (used >= maxPerGroup) {
                return false
            }
            groupCounts[key] = used + 1
            result += item
            return true
        }

        selected.forEach { item ->
            if (result.size < count) {
                tryAdd(item)
            }
        }
        if (result.size < count) {
            backup.forEach { item ->
                if (result.size < count && item !in result) {
                    tryAdd(item)
                }
            }
        }
        if (result.size < count) {
            backup.forEach { item ->
                if (result.size < count && item !in result) {
                    result += item
                }
            }
        }
        return result.take(count)
    }

    private fun selectRankBiasedIndex(
        size: Int,
        random: Random,
    ): Int {
        val totalWeight = (size.toLong() * (size + 1L)) / 2L
        var ticket = random.nextLong(totalWeight)
        for (index in 0 until size) {
            val weight = (size - index).toLong()
            if (ticket < weight) {
                return index
            }
            ticket -= weight
        }
        return size - 1
    }

    private data class TokenStats(
        val total: Int,
        val uniqueRatio: Double,
    )

    private fun tokenStats(text: String): TokenStats {
        val tokens =
            text
                .lowercase()
                .split(Regex("[^\\p{L}\\p{N}]+"))
                .filter { token -> token.length >= 3 }
        if (tokens.isEmpty()) {
            return TokenStats(total = 0, uniqueRatio = 0.0)
        }
        val unique = tokens.toSet().size
        return TokenStats(total = tokens.size, uniqueRatio = unique.toDouble() / tokens.size.toDouble())
    }

    private fun singleHopGroupKey(node: Node): String = node.getProperty("parent_document_id") ?: node.id

    private fun multiHopGroupKey(candidate: MultiHopCandidate): String {
        val leftDoc = singleHopGroupKey(candidate.left)
        val rightDoc = singleHopGroupKey(candidate.right)
        return listOf(leftDoc, rightDoc).sorted().joinToString("|")
    }

    private data class PromptPlan(
        val question: String,
        val synthesizer: String,
        val queryStyle: String,
    )

    private fun generateSingleHopSamples(nodes: List<Node>): List<TestsetSample> =
        nodes.map { node ->
            val content = node.getProperty("page_content").orEmpty().trim()
            val summary = nodeSummary(node).trim()
            val entities = nodeEntities(node)
            val primaryEntity =
                entities
                    .split(',')
                    .firstOrNull()
                    ?.trim()
                    .orEmpty()
            val topic = nodeTopic(node).trim()

            val promptPlan =
                buildSingleHopPromptPlan(
                    node = node,
                    content = content,
                    summary = summary,
                    primaryEntity = primaryEntity,
                    topic = topic,
                )

            val answer =
                if (summary.isNotBlank()) {
                    summary
                } else {
                    content.take(180)
                }
            val queryLength = inferQueryLength(promptPlan.question)

            TestsetSample(
                evalSample =
                    SingleTurnSample(
                        userInput = promptPlan.question,
                        response = answer,
                        reference = answer,
                        retrievedContexts = listOf(content),
                        referenceContexts = listOf(content),
                        queryStyle = promptPlan.queryStyle,
                        queryLength = queryLength,
                        personaName = "KotlinSynthPersona",
                    ),
                synthesizerName = promptPlan.synthesizer,
            )
        }

    private fun generateMultiHopSamples(candidates: List<MultiHopCandidate>): List<TestsetSample> =
        candidates.mapNotNull { candidate ->
            val left = candidate.left
            val right = candidate.right
            val leftText = left.getProperty("page_content").orEmpty().trim()
            val rightText = right.getProperty("page_content").orEmpty().trim()
            if (leftText.isBlank() || rightText.isBlank()) {
                return@mapNotNull null
            }

            val leftSummary = nodeSummary(left).trim()
            val rightSummary = nodeSummary(right).trim()
            val leftTopic = nodeTopic(left).trim()
            val rightTopic = nodeTopic(right).trim()
            val bridgeTopic = listOf(leftTopic, rightTopic).filter { it.isNotBlank() }.distinct().joinToString(" / ")

            val promptPlan =
                buildMultiHopPromptPlan(
                    candidate = candidate,
                    leftText = leftText,
                    rightText = rightText,
                    bridgeTopic = bridgeTopic,
                )
            val responseBase =
                listOf(leftSummary, rightSummary)
                    .filter { it.isNotBlank() }
                    .joinToString(" ")
            val response = if (responseBase.isNotBlank()) responseBase else "${leftText.take(90)} ${rightText.take(90)}"

            TestsetSample(
                evalSample =
                    SingleTurnSample(
                        userInput = promptPlan.question,
                        response = response,
                        reference = response,
                        retrievedContexts = listOf(leftText, rightText),
                        referenceContexts = listOf(leftText, rightText),
                        queryStyle = promptPlan.queryStyle,
                        queryLength = inferQueryLength(promptPlan.question),
                        personaName = "KotlinSynthPersona",
                    ),
                synthesizerName = promptPlan.synthesizer,
            )
        }

    private fun buildSingleHopPromptPlan(
        node: Node,
        content: String,
        summary: String,
        primaryEntity: String,
        topic: String,
    ): PromptPlan =
        when {
            primaryEntity.isNotBlank() -> {
                PromptPlan(
                    question =
                        "How does $primaryEntity contribute to the main point in: ${content.take(64)}? " +
                            "Answer in one concise sentence grounded in the text.",
                    synthesizer = "single_hop_entity",
                    queryStyle = "PERFECT_GRAMMAR",
                )
            }

            topic.isNotBlank() -> {
                PromptPlan(
                    question =
                        "Summarize the core idea about '$topic' from: ${content.take(64)}? " +
                            "Answer in one sentence with precise wording.",
                    synthesizer = "single_hop_topic",
                    queryStyle = "WEB_SEARCH_LIKE",
                )
            }

            summary.isNotBlank() -> {
                PromptPlan(
                    question =
                        "State the key takeaway from this chunk: ${content.take(64)}. " +
                            "Keep the answer to one sentence.",
                    synthesizer = "single_hop_summary",
                    queryStyle = "PERFECT_GRAMMAR",
                )
            }

            node.type == NodeType.CHUNK -> {
                PromptPlan(
                    question =
                        "What is the key point of: ${content.take(70)}? " +
                            "Return one concise sentence.",
                    synthesizer = "single_hop_chunk",
                    queryStyle = "WEB_SEARCH_LIKE",
                )
            }

            else -> {
                PromptPlan(
                    question =
                        "What is the key point of: ${content.take(70)}? " +
                            "Return one concise sentence.",
                    synthesizer = "single_hop_stub",
                    queryStyle = "PERFECT_GRAMMAR",
                )
            }
        }

    private fun buildMultiHopPromptPlan(
        candidate: MultiHopCandidate,
        leftText: String,
        rightText: String,
        bridgeTopic: String,
    ): PromptPlan =
        if (candidate.relationship.type == "semantic_overlap") {
            PromptPlan(
                question =
                    "Synthesize the shared theme (${if (bridgeTopic.isBlank()) "common topic" else bridgeTopic}) " +
                        "between \"${leftText.take(30)}\" and \"${rightText.take(30)}\". " +
                        "Answer in two concise sentences grounded in both contexts.",
                synthesizer = "multi_hop_overlap",
                queryStyle = "PERFECT_GRAMMAR",
            )
        } else {
            PromptPlan(
                question =
                    "Explain the progression from \"${leftText.take(30)}\" to \"${rightText.take(30)}\". " +
                        "Answer in two concise sentences using both contexts.",
                synthesizer = "multi_hop_sequence",
                queryStyle = "WEB_SEARCH_LIKE",
            )
        }

    private fun inferQueryLength(question: String): String {
        val words = question.split(Regex("\\s+")).count { token -> token.isNotBlank() }
        return when {
            words <= 8 -> "SHORT"
            words <= 16 -> "MEDIUM"
            else -> "LONG"
        }
    }

    private fun nodeSummary(node: Node): String =
        node
            .getProperty(PropertyNames.SUMMARY_LLM_BASED)
            ?.takeIf { value -> value.isNotBlank() }
            ?: node.getProperty("source_document_summary").orEmpty()

    private fun nodeEntities(node: Node): String =
        node
            .getProperty(PropertyNames.ENTITIES_REGEX)
            ?.takeIf { value -> value.isNotBlank() }
            ?: node.getProperty("source_document_entities").orEmpty()

    private fun nodeTopic(node: Node): String =
        node
            .getProperty(PropertyNames.EMBEDDING_TOPIC_TAG)
            ?.takeIf { value -> value.isNotBlank() }
            ?: node.getProperty("source_document_topic").orEmpty()
}

/**
 * Returns a no-op transform sequence used as the default preprocessing pipeline.
 *
 * @return Empty sequence of transforms.
 */
fun emptyTransforms(): Transforms = SequenceTransforms(emptyList())
