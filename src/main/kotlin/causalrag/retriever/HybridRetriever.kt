package causalrag.retriever

import causalrag.causalgraph.retriever.CausalPathRetriever
import io.github.oshai.kotlinlogging.KotlinLogging
import java.util.Collections
import java.util.LinkedHashMap

private val logger = KotlinLogging.logger {}

/**
 * Combines semantic, causal, and optional BM25 retrieval signals.
 *
 * @param vectorRetriever Primary semantic retriever used to fetch candidate passages.
 * @param graphRetriever Retriever that provides causal nodes and paths for query-aware reranking.
 * @param semanticWeight Weight assigned to the semantic retrieval score before normalization.
 * @param causalWeight Weight assigned to the causal matching score before normalization.
 * @param bm25Weight Weight assigned to the BM25 keyword score before normalization.
 * @param bm25Retriever Optional BM25 retriever used when lexical scoring is enabled.
 * @param rerankingFactor Multiplier controlling how many semantic candidates are fetched before reranking.
 * @param minCausalMatches Minimum number of matched causal nodes required for a passage to survive filtering.
 * @param dynamicWeightingEnabled Enables query-aware semantic/causal weight gating.
 * @param gatingPathCoef Coefficient `a` applied to normalized path count in the gating logit.
 * @param gatingCoverageCoef Coefficient `b` applied to candidate node coverage in the gating logit.
 * @param gatingAmbiguityCoef Coefficient `c` applied to estimated query ambiguity in the gating logit.
 * @param gatingBias Bias term added to the gating logit.
 * @param gatingPathScale Normalization scale used for causal path count.
 * @param twoPassAdaptiveEnabled Enables two-pass adaptive weighting using causal agreement of first-pass candidates.
 * @param twoPassSemanticPrior Semantic-heavy prior used during first-pass ranking (0.0-1.0).
 * @param twoPassAgreementBlend Blend factor when moving from current causal weight to agreement-derived causal weight.
 * @param twoPassTopKMultiplier Candidate count multiplier used when computing first-pass agreement.
 * @param confidenceBasedSwitchEnabled Enables confidence-based fallback toward semantic weighting.
 * @param confidenceNodeTarget Target node count used to normalize causal node confidence.
 * @param confidencePathTarget Target path count used to normalize causal path confidence.
 * @param cacheResults Whether query results should be cached in memory.
 * @param cacheMaxEntries Maximum number of cached query entries when caching is enabled.
 */
@Suppress("TooGenericExceptionCaught")
class HybridRetriever(
    private val semanticRetriever: SemanticRetriever,
    private val graphRetriever: CausalPathRetriever,
    private var semanticWeight: Double = 0.4,
    private var causalWeight: Double = 0.6,
    private var bm25Weight: Double = 0.0,
    private val bm25Retriever: Bm25Retriever? = null,
    private val rerankingFactor: Int = 2,
    private val minCausalMatches: Int = 1,
    private val dynamicWeightingEnabled: Boolean = false,
    private val gatingPathCoef: Double = 2.0,
    private val gatingCoverageCoef: Double = 2.0,
    private val gatingAmbiguityCoef: Double = 1.5,
    private val gatingBias: Double = -0.5,
    private val gatingPathScale: Double = 3.0,
    private val twoPassAdaptiveEnabled: Boolean = false,
    private val twoPassSemanticPrior: Double = 0.85,
    private val twoPassAgreementBlend: Double = 0.7,
    private val twoPassTopKMultiplier: Int = 2,
    private val confidenceBasedSwitchEnabled: Boolean = false,
    private val confidenceNodeTarget: Double = 6.0,
    private val confidencePathTarget: Double = 3.0,
    private val cacheResults: Boolean = true,
    private val cacheMaxEntries: Int = 1000,
) {
    private data class QueryWeights(
        val semantic: Double,
        val causal: Double,
        val bm25: Double,
        val diagnostics: Map<String, Any>,
    )

    private val queryCache: MutableMap<String, List<Map<String, Any>>> =
        if (cacheResults) {
            Collections.synchronizedMap(
                object : LinkedHashMap<String, List<Map<String, Any>>>(16, 0.75f, true) {
                    override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, List<Map<String, Any>>>?): Boolean =
                        size > cacheMaxEntries
                },
            )
        } else {
            mutableMapOf()
        }

    init {
        if (cacheResults) {
            require(cacheMaxEntries > 0) { "cacheMaxEntries must be positive when cacheResults is enabled." }
        }
        require(gatingPathScale > 0.0) { "gatingPathScale must be positive." }
        require(twoPassSemanticPrior in 0.0..1.0) { "twoPassSemanticPrior must be within [0, 1]." }
        require(twoPassAgreementBlend in 0.0..1.0) { "twoPassAgreementBlend must be within [0, 1]." }
        require(twoPassTopKMultiplier > 0) { "twoPassTopKMultiplier must be positive." }
        require(confidenceNodeTarget > 0.0) { "confidenceNodeTarget must be positive." }
        require(confidencePathTarget > 0.0) { "confidencePathTarget must be positive." }
        val total = semanticWeight + causalWeight + bm25Weight
        require(total > 0.0) { "Sum of semanticWeight, causalWeight, and bm25Weight must be positive." }
        if (kotlin.math.abs(total - 1.0) > 1e-9) {
            semanticWeight /= total
            causalWeight /= total
            if (bm25Weight > 0.0) {
                bm25Weight /= total
            }
        }
    }

    private fun scorePassage(
        passage: String,
        pathNodes: List<String>,
        causalPaths: List<List<String>>,
        semanticScore: Double = 0.0,
        bm25Score: Double = 0.0,
        semanticBlendWeight: Double = semanticWeight,
        causalBlendWeight: Double = causalWeight,
        bm25BlendWeight: Double = bm25Weight,
    ): Pair<Double, Map<String, Any>> {
        val passageLower = passage.lowercase()
        val matchedNodes = pathNodes.filter { passageLower.contains(it.lowercase()) }
        val nodeMatchScore = matchedNodes.size.toDouble() / maxOf(pathNodes.size, 1)

        val pathMatches = mutableListOf<Pair<String, String>>()
        var totalPairs = 0
        for (path in causalPaths) {
            if (path.size < 2) continue
            for (i in 0 until path.size - 1) {
                val cause = path[i].lowercase()
                val effect = path[i + 1].lowercase()
                totalPairs += 1
                if (passageLower.contains(cause) && passageLower.contains(effect)) {
                    val causePos = passageLower.indexOf(cause)
                    val effectPos = passageLower.indexOf(effect)
                    // Heuristic assumes cause appears before effect; reversed phrasing may be missed.
                    if (causePos < effectPos) {
                        pathMatches.add(cause to effect)
                    }
                }
            }
        }
        val overallPathScore = if (totalPairs > 0) pathMatches.size.toDouble() / totalPairs else 0.0
        val causalScore = 0.7 * nodeMatchScore + 0.3 * overallPathScore
        val combinedScore = semanticBlendWeight * semanticScore + causalBlendWeight * causalScore
        val finalScore = combinedScore + bm25BlendWeight * bm25Score

        return finalScore to
            mapOf(
                "matched_nodes" to matchedNodes,
                "node_score" to nodeMatchScore,
                "path_matches" to pathMatches,
                "path_score" to overallPathScore,
                "causal_score" to causalScore,
                "semantic_score" to semanticScore,
                "bm25_score" to bm25Score,
                "semantic_weight" to semanticBlendWeight,
                "causal_weight" to causalBlendWeight,
                "bm25_weight" to bm25BlendWeight,
                "combined_score" to finalScore,
            )
    }

    /**
     * Retrieves the top passages for a query.
     *
     * @param query User query.
     * @param topK Maximum number of passages to return.
     * @return Retrieved passages ordered by hybrid score.
     */
    fun retrieve(
        query: String,
        topK: Int = 5,
    ): List<String> = retrieveWithDetails(query, topK).map { it["passage"] as String }

    /**
     * Retrieves passages together with hybrid scores.
     *
     * @param query User query.
     * @param topK Maximum number of passages to return.
     * @return Passage-score pairs.
     */
    fun retrieveWithScores(
        query: String,
        topK: Int = 5,
    ): List<Pair<String, Double>> = retrieveWithDetails(query, topK).map { it["passage"] as String to (it["score"] as Double) }

    /**
     * Retrieves passages with full scoring details.
     *
     * @param query User query.
     * @param topK Maximum number of passages to return.
     * @return Result maps containing passage text, score, and feature breakdowns.
     */
    fun retrieveWithDetails(
        query: String,
        topK: Int = 5,
    ): List<Map<String, Any>> {
        if (cacheResults) {
            val cached = queryCache[query]
            if (cached != null) {
                return cached.take(topK)
            }
        }

        val expandedK = topK * rerankingFactor
        val semanticResults =
            try {
                semanticRetriever.searchWithScores(query, topK = expandedK)
            } catch (ex: Exception) {
                logger.error(ex) { "Error retrieving vector results" }
                emptyList()
            }

        val bm25Scores: Map<String, Double> =
            if (bm25Retriever != null && bm25Weight > 0.0) {
                try {
                    val bm25Results = bm25Retriever.retrieve(query, topK = expandedK)
                    bm25Results.associate { result ->
                        val passage = result["passage"] as String
                        val score = result["score"] as Double
                        passage to score
                    }
                } catch (ex: Exception) {
                    logger.error(ex) { "Error retrieving BM25 results" }
                    emptyMap()
                }
            } else {
                emptyMap()
            }

        val candidates = linkedMapOf<String, Double>()
        semanticResults.forEach { (passage, semanticScore) ->
            candidates[passage] = semanticScore
        }
        bm25Scores.keys.forEach { passage ->
            candidates.putIfAbsent(passage, 0.0)
        }

        if (candidates.isEmpty()) {
            logger.warn { "No retrieval results found for query" }
            return emptyList()
        }

        val pathNodes =
            try {
                graphRetriever.retrievePathNodes(query)
            } catch (ex: Exception) {
                logger.error(ex) { "Error retrieving causal nodes" }
                emptyList()
            }
        val causalPaths =
            try {
                graphRetriever.retrievePaths(query, maxPaths = 3)
            } catch (ex: Exception) {
                logger.error(ex) { "Error retrieving causal paths" }
                emptyList()
            }
        var queryWeights = resolveQueryWeights(query, pathNodes, causalPaths, candidates.keys)
        queryWeights = applyConfidenceBasedSwitch(queryWeights, pathNodes, causalPaths)
        queryWeights =
            applyTwoPassAdaptiveWeights(
                current = queryWeights,
                topK = topK,
                candidates = candidates,
                bm25Scores = bm25Scores,
                pathNodes = pathNodes,
                causalPaths = causalPaths,
            )

        val scoredResults = mutableListOf<Map<String, Any>>()
        for ((passage, semanticScore) in candidates) {
            val keywordScore = bm25Scores[passage] ?: 0.0
            val (score, details) =
                scorePassage(
                    passage = passage,
                    pathNodes = pathNodes,
                    causalPaths = causalPaths,
                    semanticScore = semanticScore,
                    bm25Score = keywordScore,
                    semanticBlendWeight = queryWeights.semantic,
                    causalBlendWeight = queryWeights.causal,
                    bm25BlendWeight = queryWeights.bm25,
                )
            val enrichedDetails =
                details +
                    mapOf(
                        "dynamic_weighting_enabled" to dynamicWeightingEnabled,
                        "weight_diagnostics" to queryWeights.diagnostics,
                    )
            val matchedNodes = details["matched_nodes"] as? List<*> ?: emptyList<Any>()
            if (matchedNodes.size < minCausalMatches && pathNodes.isNotEmpty()) {
                continue
            }
            scoredResults.add(
                mapOf(
                    "passage" to passage,
                    "score" to score,
                    "details" to enrichedDetails,
                ),
            )
        }

        val sorted = scoredResults.sortedByDescending { it["score"] as Double }
        if (cacheResults) {
            queryCache[query] = sorted
        }
        return sorted.take(topK)
    }

    /**
     * Explains how the hybrid retriever scored a passage for a query.
     *
     * Detailed explanations are only available after [retrieveWithDetails], [retrieveWithScores], or [retrieve]
     * has populated the internal cache for the same query. When caching is disabled, this method always falls back
     * to a generic explanation.
     *
     * @param query User query.
     * @param passage Passage to explain.
     * @return Human-readable explanation string.
     */
    fun getExplanation(
        query: String,
        passage: String,
    ): String {
        val fallbackMessage =
            "This passage was retrieved as relevant to the query: $query. " +
                "No detailed scoring information is available."
        val cached = queryCache[query] ?: return fallbackMessage
        val result =
            cached.firstOrNull { it["passage"] == passage }
                ?: return fallbackMessage
        val details = result["details"] as Map<*, *>
        val explanation = mutableListOf("Hybrid retrieval explanation for: $query")
        val semanticScore = details["semantic_score"] as? Double ?: 0.0
        val causalScore = details["causal_score"] as? Double ?: 0.0
        val combinedScore = details["combined_score"] as? Double ?: 0.0
        val bm25Score = details["bm25_score"] as? Double ?: 0.0
        val semanticWeightUsed = details["semantic_weight"] as? Double ?: semanticWeight
        val causalWeightUsed = details["causal_weight"] as? Double ?: causalWeight
        val bm25WeightUsed = details["bm25_weight"] as? Double ?: bm25Weight
        explanation.add("\nSemantic relevance score: ${"%.2f".format(semanticScore)} (weight: ${"%.2f".format(semanticWeightUsed)})")
        explanation.add("\nCausal relevance score: ${"%.2f".format(causalScore)} (weight: ${"%.2f".format(causalWeightUsed)})")
        if (bm25WeightUsed > 0.0) {
            explanation.add("\nBM25 relevance score: ${"%.2f".format(bm25Score)} (weight: ${"%.2f".format(bm25WeightUsed)})")
        }
        val matchedNodes = details["matched_nodes"] as? List<*> ?: emptyList<Any>()
        if (matchedNodes.isNotEmpty()) {
            explanation.add("\nMatched causal concepts (${matchedNodes.size} concepts):")
            matchedNodes.forEach { explanation.add("- $it") }
        }
        val pathMatches = details["path_matches"] as? List<*> ?: emptyList<Any>()
        if (pathMatches.isNotEmpty()) {
            explanation.add("\nPreserved causal relationships:")
            pathMatches.forEach { explanation.add("- $it") }
        }
        explanation.add("\nOverall score: ${"%.2f".format(combinedScore)}")
        return explanation.joinToString("\n")
    }

    /**
     * Clears the query result cache.
     */
    fun clearCache() {
        queryCache.clear()
    }

    private fun resolveQueryWeights(
        query: String,
        pathNodes: List<String>,
        causalPaths: List<List<String>>,
        candidatePassages: Set<String>,
    ): QueryWeights {
        if (!dynamicWeightingEnabled) {
            return QueryWeights(
                semantic = semanticWeight,
                causal = causalWeight,
                bm25 = bm25Weight,
                diagnostics = mapOf("mode" to "static"),
            )
        }

        val pathCountNorm = (causalPaths.size.toDouble() / gatingPathScale).coerceIn(0.0, 1.0)
        val nodeCoverage = calculateCandidateNodeCoverage(pathNodes, candidatePassages)
        val ambiguity = estimateQueryAmbiguity(query)
        val logit =
            (gatingPathCoef * pathCountNorm) +
                (gatingCoverageCoef * nodeCoverage) -
                (gatingAmbiguityCoef * ambiguity) +
                gatingBias
        val causalGate = sigmoid(logit)
        val nonBm25Budget = (1.0 - bm25Weight).coerceIn(0.0, 1.0)
        val dynamicCausalWeight = nonBm25Budget * causalGate
        val dynamicSemanticWeight = nonBm25Budget * (1.0 - causalGate)

        return QueryWeights(
            semantic = dynamicSemanticWeight,
            causal = dynamicCausalWeight,
            bm25 = bm25Weight,
            diagnostics =
                mapOf(
                    "mode" to "dynamic",
                    "path_count" to causalPaths.size,
                    "path_count_norm" to pathCountNorm,
                    "node_coverage" to nodeCoverage,
                    "ambiguity" to ambiguity,
                    "causal_gate" to causalGate,
                ),
        )
    }

    private fun applyConfidenceBasedSwitch(
        current: QueryWeights,
        pathNodes: List<String>,
        causalPaths: List<List<String>>,
    ): QueryWeights {
        if (!confidenceBasedSwitchEnabled) return current
        val nodeConfidence = (pathNodes.size.toDouble() / confidenceNodeTarget).coerceIn(0.0, 1.0)
        val pathConfidence = (causalPaths.size.toDouble() / confidencePathTarget).coerceIn(0.0, 1.0)
        val causalConfidence = (nodeConfidence + pathConfidence) / 2.0
        val nonBm25Budget = (1.0 - current.bm25).coerceIn(0.0, 1.0)
        val adjustedCausal = (current.causal * causalConfidence).coerceIn(0.0, nonBm25Budget)
        val adjustedSemantic = nonBm25Budget - adjustedCausal

        return current.copy(
            semantic = adjustedSemantic,
            causal = adjustedCausal,
            diagnostics =
                current.diagnostics +
                    mapOf(
                        "confidence_switch" to "enabled",
                        "causal_node_confidence" to nodeConfidence,
                        "causal_path_confidence" to pathConfidence,
                        "causal_confidence" to causalConfidence,
                    ),
        )
    }

    private fun applyTwoPassAdaptiveWeights(
        current: QueryWeights,
        topK: Int,
        candidates: Map<String, Double>,
        bm25Scores: Map<String, Double>,
        pathNodes: List<String>,
        causalPaths: List<List<String>>,
    ): QueryWeights {
        if (!twoPassAdaptiveEnabled || candidates.isEmpty()) return current
        val nonBm25Budget = (1.0 - current.bm25).coerceIn(0.0, 1.0)
        val firstPassSemantic = nonBm25Budget * twoPassSemanticPrior
        val firstPassCausal = nonBm25Budget - firstPassSemantic
        val firstPassTopN = maxOf(1, topK * twoPassTopKMultiplier)

        val firstPassRanked =
            candidates
                .map { (passage, semanticScore) ->
                    val bm25Score = bm25Scores[passage] ?: 0.0
                    val (score, details) =
                        scorePassage(
                            passage = passage,
                            pathNodes = pathNodes,
                            causalPaths = causalPaths,
                            semanticScore = semanticScore,
                            bm25Score = bm25Score,
                            semanticBlendWeight = firstPassSemantic,
                            causalBlendWeight = firstPassCausal,
                            bm25BlendWeight = current.bm25,
                        )
                    Triple(
                        passage,
                        score,
                        (details["causal_score"] as? Double ?: 0.0).coerceIn(0.0, 1.0),
                    )
                }.sortedByDescending { it.second }
                .take(firstPassTopN)

        val causalAgreement =
            if (firstPassRanked.isEmpty()) {
                0.0
            } else {
                firstPassRanked
                    .map { (_, _, causalScore) -> causalScore }
                    .average()
                    .coerceIn(0.0, 1.0)
            }
        val targetCausal = nonBm25Budget * causalAgreement
        val adaptedCausal = lerp(current.causal, targetCausal, twoPassAgreementBlend).coerceIn(0.0, nonBm25Budget)
        val adaptedSemantic = nonBm25Budget - adaptedCausal

        return current.copy(
            semantic = adaptedSemantic,
            causal = adaptedCausal,
            diagnostics =
                current.diagnostics +
                    mapOf(
                        "two_pass_adaptive" to "enabled",
                        "two_pass_first_pass_top_n" to firstPassRanked.size,
                        "two_pass_causal_agreement" to causalAgreement,
                        "two_pass_target_causal_weight" to targetCausal,
                    ),
        )
    }

    private fun lerp(
        from: Double,
        to: Double,
        t: Double,
    ): Double = from + (to - from) * t

    private fun calculateCandidateNodeCoverage(
        pathNodes: List<String>,
        candidatePassages: Set<String>,
    ): Double {
        if (pathNodes.isEmpty() || candidatePassages.isEmpty()) return 0.0
        val lowerPassages = candidatePassages.map { it.lowercase() }
        val coveredNodes =
            pathNodes.count { node ->
                val nodeLower = node.lowercase()
                lowerPassages.any { passage -> passage.contains(nodeLower) }
            }
        return coveredNodes.toDouble() / pathNodes.size.toDouble()
    }

    private fun estimateQueryAmbiguity(query: String): Double {
        val nodeScores = graphRetriever.retrieveNodes(query, topK = 3, threshold = 0.0)
        if (nodeScores.isEmpty()) return 1.0
        val top1 = nodeScores.first().second.coerceAtLeast(0.0)
        if (nodeScores.size == 1) return (1.0 - top1).coerceIn(0.0, 1.0)
        val top2 = nodeScores[1].second.coerceAtLeast(0.0)
        if (top1 <= 1e-9) return 1.0
        val margin = ((top1 - top2) / top1).coerceIn(0.0, 1.0)
        return 1.0 - margin
    }

    private fun sigmoid(value: Double): Double {
        val capped = value.coerceIn(-40.0, 40.0)
        return 1.0 / (1.0 + kotlin.math.exp(-capped))
    }
}
