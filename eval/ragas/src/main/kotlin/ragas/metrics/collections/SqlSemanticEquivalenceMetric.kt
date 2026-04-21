package ragas.metrics.collections

import ragas.metrics.BaseMetric
import ragas.metrics.MetricOutputType
import ragas.metrics.MetricType
import ragas.metrics.SingleTurnMetric
import ragas.model.SingleTurnSample

/**
 * Implements [SqlSemanticEquivalenceMetric].
 */
class SqlSemanticEquivalenceMetric(
    name: String = "sql_semantic_equivalence",
) : BaseMetric(
        name = name,
        requiredColumns = mapOf(MetricType.SINGLE_TURN to setOf("response", "reference")),
        outputType = MetricOutputType.BINARY,
    ),
    SingleTurnMetric {
    /**
     * Executes singleTurnAscore.
     * @param sample Evaluation sample to score.
     */
    override suspend fun singleTurnAscore(sample: SingleTurnSample): Any {
        val response = sample.response.orEmpty().trim()
        val reference = sample.reference.orEmpty().trim()
        require(response.isNotBlank()) { "response must be a non-empty SQL query string" }
        require(reference.isNotBlank()) { "reference must be a non-empty SQL query string" }

        val responseQuery = parseSql(response)
        val referenceQuery = parseSql(reference)
        val score = semanticSimilarityScore(responseQuery, referenceQuery)
        return if (score >= EQUIVALENCE_THRESHOLD) 1.0 else 0.0
    }

    private fun semanticSimilarityScore(
        left: ParsedSql,
        right: ParsedSql,
    ): Double {
        var score = 1.0

        if (left.fromTable != right.fromTable) {
            score -= 0.45
        }

        if (left.selectColumns != right.selectColumns) {
            score -= 0.2
        }

        if (left.groupByColumns != right.groupByColumns) {
            score -= 0.15
        }

        if (left.orderByColumns != right.orderByColumns) {
            score -= 0.1
        }

        if (left.whereNormalized != right.whereNormalized) {
            val tokenOverlap = tokenJaccard(left.whereTokens, right.whereTokens)
            score -= (0.25 * (1.0 - tokenOverlap))
        }

        if (left.aggregateFunctions != right.aggregateFunctions) {
            score -= 0.35
        }

        return score.coerceIn(0.0, 1.0)
    }

    private fun tokenJaccard(
        left: Set<String>,
        right: Set<String>,
    ): Double {
        if (left.isEmpty() && right.isEmpty()) {
            return 1.0
        }
        if (left.isEmpty() || right.isEmpty()) {
            return 0.0
        }
        val intersection = left.intersect(right).size.toDouble()
        val union = left.union(right).size.toDouble()
        return intersection / union
    }

    private fun parseSql(sql: String): ParsedSql {
        val normalized = normalizeSql(sql)
        val selectColumns = extractColumns(normalized, SELECT_PATTERN)
        val fromTable =
            FROM_PATTERN
                .find(normalized)
                ?.groupValues
                ?.get(1)
                .orEmpty()
        val whereRaw =
            WHERE_PATTERN
                .find(normalized)
                ?.groupValues
                ?.get(1)
                .orEmpty()
        val groupByColumns = extractColumns(normalized, GROUP_BY_PATTERN)
        val orderByColumns = extractColumns(normalized, ORDER_BY_PATTERN)
        val aggregateFunctions = AGG_PATTERN.findAll(normalized).map { it.groupValues[1] }.toSet()

        val whereNormalized = normalizeWhere(whereRaw)
        val whereTokens = TOKEN_PATTERN.findAll(whereNormalized).map { it.value }.toSet()
        return ParsedSql(
            selectColumns = selectColumns,
            fromTable = fromTable,
            whereNormalized = whereNormalized,
            whereTokens = whereTokens,
            groupByColumns = groupByColumns,
            orderByColumns = orderByColumns,
            aggregateFunctions = aggregateFunctions,
        )
    }

    private fun normalizeSql(sql: String): String =
        sql
            .lowercase()
            .replace(Regex("\\s+"), " ")
            .replace(TRUE_LITERAL_REGEX, "1")
            .replace(FALSE_LITERAL_REGEX, "0")
            .trim()

    private fun extractColumns(
        sql: String,
        pattern: Regex,
    ): Set<String> {
        val raw =
            pattern
                .find(sql)
                ?.groupValues
                ?.get(1)
                .orEmpty()
        if (raw.isBlank()) {
            return emptySet()
        }
        return raw
            .split(',')
            .map { token ->
                token
                    .replace(Regex("\\s+as\\s+\\w+"), "")
                    .replace(Regex("\\s+"), " ")
                    .trim()
            }.filter { it.isNotBlank() }
            .toSet()
    }

    private fun normalizeWhere(whereClause: String): String =
        whereClause
            .replace(Regex("\\s+"), " ")
            .replace(TRUE_LITERAL_REGEX, "1")
            .replace(FALSE_LITERAL_REGEX, "0")
            .replace(" = 1", " =1")
            .replace(" = 0", " =0")
            .trim()

    private data class ParsedSql(
        val selectColumns: Set<String>,
        val fromTable: String,
        val whereNormalized: String,
        val whereTokens: Set<String>,
        val groupByColumns: Set<String>,
        val orderByColumns: Set<String>,
        val aggregateFunctions: Set<String>,
    )

    private companion object {
        val SELECT_PATTERN = Regex("\\bselect\\b\\s+(.*?)\\s+\\bfrom\\b")
        val FROM_PATTERN = Regex("\\bfrom\\b\\s+([a-zA-Z0-9_.]+)")
        val WHERE_PATTERN = Regex("\\bwhere\\b\\s+(.*?)(?:\\bgroup\\s+by\\b|\\border\\s+by\\b|\\blimit\\b|$)")
        val GROUP_BY_PATTERN = Regex("\\bgroup\\s+by\\b\\s+(.*?)(?:\\border\\s+by\\b|\\blimit\\b|$)")
        val ORDER_BY_PATTERN = Regex("\\border\\s+by\\b\\s+(.*?)(?:\\blimit\\b|$)")
        val AGG_PATTERN = Regex("\\b(sum|count|avg|min|max)\\s*\\(")
        val TOKEN_PATTERN = Regex("[a-zA-Z0-9_]+")
        val TRUE_LITERAL_REGEX = Regex("\\btrue\\b")
        val FALSE_LITERAL_REGEX = Regex("\\bfalse\\b")
        const val EQUIVALENCE_THRESHOLD = 0.75
    }
}
