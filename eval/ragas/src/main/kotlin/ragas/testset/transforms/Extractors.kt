package ragas.testset.transforms

import ragas.testset.graph.Node
import kotlin.math.min

/**
 * Extractor that derives entity strings from node text using a regex pattern.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate.
 * @property sourceProperty Source property key.
 * @property targetProperty Target property key.
 * @property maxEntities Maximum extracted entities.
 */
class RegexEntityExtractor(
    override val name: String = "regex_entity_extractor",
    override val filterNodes: (Node) -> Boolean = defaultFilter,
    private val sourceProperty: String = "page_content",
    private val targetProperty: String = PropertyNames.ENTITIES_REGEX,
    private val maxEntities: Int = 8,
) : Extractor(name = name, filterNodes = filterNodes) {
    private val entityPattern = Regex("\\b[A-Z][a-zA-Z]{2,}\\b")

    /**
     * Returns the property key written by this extractor.
     *
     * @param node Node to extract from.
     * @return Target property name.
     */
    override fun propertyName(node: Node): String = targetProperty

    /**
     * Extracts one property value from a node.
     *
     * @param node Node to extract from.
     * @return Extracted entity list.
     */
    override suspend fun extract(node: Node): String {
        val text = node.getProperty(sourceProperty).orEmpty()
        if (text.isBlank()) {
            return ""
        }

        val entities =
            entityPattern
                .findAll(text)
                .map { match -> match.value }
                .distinct()
                .take(maxEntities)
                .toList()

        return entities.joinToString(", ")
    }
}

/**
 * Extractor that derives a deterministic topic token from node text as a lightweight stand-in for embedding-based topics.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate.
 * @property sourceProperty Source property key.
 * @property targetProperty Target property key.
 */
class EmbeddingsTopicExtractor(
    override val name: String = "embeddings_topic_extractor",
    override val filterNodes: (Node) -> Boolean = defaultFilter,
    private val sourceProperty: String = "page_content",
    private val targetProperty: String = PropertyNames.EMBEDDING_TOPIC_TAG,
) : Extractor(name = name, filterNodes = filterNodes) {
    /**
     * Returns the property key written by this extractor.
     *
     * @param node Node to extract from.
     * @return Target property name.
     */
    override fun propertyName(node: Node): String = targetProperty

    /**
     * Extracts one property value from a node.
     *
     * @param node Node to extract from.
     * @return Extracted topic token.
     */
    override suspend fun extract(node: Node): String {
        val text = node.getProperty(sourceProperty).orEmpty()
        val token =
            text
                .lowercase()
                .split(Regex("[^a-z0-9]+"))
                .filter { word -> word.length >= 4 }
                .groupingBy { word -> word }
                .eachCount()
                .entries
                .sortedWith(compareByDescending<Map.Entry<String, Int>> { it.value }.thenByDescending { it.key.length })
                .firstOrNull()
                ?.key
                .orEmpty()

        return token
    }
}

/**
 * Extractor that derives a short summary from node text using sentence-level clipping heuristics.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate.
 * @property sourceProperty Source property key.
 * @property targetProperty Target property key.
 * @property maxWords Maximum summary words.
 */
class LlmBasedSummaryExtractor(
    override val name: String = "llm_based_summary_extractor",
    override val filterNodes: (Node) -> Boolean = defaultFilter,
    private val sourceProperty: String = "page_content",
    private val targetProperty: String = PropertyNames.SUMMARY_LLM_BASED,
    private val maxWords: Int = 24,
) : Extractor(name = name, filterNodes = filterNodes) {
    /**
     * Returns the property key written by this extractor.
     *
     * @param node Node to extract from.
     * @return Target property name.
     */
    override fun propertyName(node: Node): String = targetProperty

    /**
     * Extracts one property value from a node.
     *
     * @param node Node to extract from.
     * @return Extracted summary text.
     */
    override suspend fun extract(node: Node): String {
        val text = node.getProperty(sourceProperty).orEmpty().trim()
        if (text.isBlank()) {
            return ""
        }

        val sentences =
            text
                .split(Regex("(?<=[.!?])\\s+"))
                .map { sentence -> sentence.trim() }
                .filter { sentence -> sentence.isNotBlank() }

        val summary = sentences.take(2).joinToString(" ")
        val words = summary.split(Regex("\\s+")).filter { word -> word.isNotBlank() }
        val clipped = words.take(min(words.size, maxWords)).joinToString(" ")
        return clipped
    }
}
