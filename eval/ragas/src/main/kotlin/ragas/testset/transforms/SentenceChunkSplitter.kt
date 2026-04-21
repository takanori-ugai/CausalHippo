package ragas.testset.transforms

import ragas.testset.graph.Node
import ragas.testset.graph.NodeType
import ragas.testset.graph.Relationship
import java.text.BreakIterator
import java.util.Locale

/**
 * Splits document text into sentence-based chunk nodes and links them with chunk relationships.
 *
 * @property name Transformation name.
 * @property filterNodes Node selection predicate for source documents.
 * @property sourceProperty Source property key.
 * @property maxSentencesPerChunk Maximum sentences per chunk.
 */
class SentenceChunkSplitter(
    override val name: String = "sentence_chunk_splitter",
    override val filterNodes: (Node) -> Boolean = { node -> node.type == NodeType.DOCUMENT },
    private val sourceProperty: String = "page_content",
    private val maxSentencesPerChunk: Int = 1,
) : Splitter(name = name, filterNodes = filterNodes) {
    init {
        require(maxSentencesPerChunk > 0) { "maxSentencesPerChunk must be > 0" }
    }

    /**
     * Splits one source node into chunk nodes and intra-document relationships.
     *
     * @param node Source document node.
     * @return Pair of generated chunk nodes and `child` relationships.
     */
    override suspend fun split(node: Node): Pair<List<Node>, List<Relationship>> {
        val text = node.getProperty(sourceProperty).orEmpty().trim()
        if (text.isBlank()) {
            return emptyList<Node>() to emptyList()
        }

        val sentences =
            splitSentences(text)

        if (sentences.isEmpty()) {
            return emptyList<Node>() to emptyList()
        }

        val grouped = sentences.chunked(maxSentencesPerChunk)
        val chunks =
            grouped.mapIndexed { index, parts ->
                val chunkText = parts.joinToString(" ").trim()
                Node(
                    type = NodeType.CHUNK,
                    properties =
                        mutableMapOf(
                            "page_content" to chunkText,
                            "chunk_index" to index.toString(),
                            "parent_document_id" to node.id,
                            "source_document_summary" to node.getProperty(PropertyNames.SUMMARY_LLM_BASED).orEmpty(),
                            "source_document_entities" to node.getProperty(PropertyNames.ENTITIES_REGEX).orEmpty(),
                            "source_document_topic" to node.getProperty(PropertyNames.EMBEDDING_TOPIC_TAG).orEmpty(),
                        ),
                )
            }

        val relationships =
            chunks.map { chunk ->
                Relationship(
                    type = "child",
                    sourceId = node.id,
                    targetId = chunk.id,
                )
            }

        return chunks to relationships
    }

    private fun splitSentences(text: String): List<String> {
        val iterator = BreakIterator.getSentenceInstance(Locale.ROOT)
        iterator.setText(text)
        val rawSentences = mutableListOf<String>()
        var start = iterator.first()
        var end = iterator.next()
        while (end != BreakIterator.DONE) {
            val sentence = text.substring(start, end).trim()
            if (sentence.isNotBlank()) {
                rawSentences += sentence
            }
            start = end
            end = iterator.next()
        }
        if (rawSentences.isEmpty()) {
            return emptyList()
        }
        return mergeLikelyFalseSplits(rawSentences)
    }

    private fun mergeLikelyFalseSplits(sentences: List<String>): List<String> {
        val merged = mutableListOf<String>()
        var index = 0
        while (index < sentences.size) {
            var current = sentences[index]
            while (index + 1 < sentences.size && shouldMergeWithNext(current, sentences[index + 1])) {
                current = "$current ${sentences[index + 1]}"
                index += 1
            }
            merged += current
            index += 1
        }
        return merged
    }

    private fun shouldMergeWithNext(
        current: String,
        next: String,
    ): Boolean {
        val trimmedCurrent = current.trim()
        val trimmedNext = next.trim()
        if (trimmedCurrent.isEmpty() || trimmedNext.isEmpty()) {
            return false
        }

        if (trimmedCurrent in knownAbbreviations || hasTrailingKnownAbbreviation(trimmedCurrent)) {
            return true
        }

        val nextFirstChar = trimmedNext.first()
        if (nextFirstChar.isLowerCase()) {
            return true
        }

        if (decimalTailRegex.containsMatchIn(trimmedCurrent) && nextFirstChar.isDigit()) {
            return true
        }

        return false
    }

    companion object {
        private val knownAbbreviations =
            setOf(
                "Mr.",
                "Mrs.",
                "Ms.",
                "Dr.",
                "Prof.",
                "Sr.",
                "Jr.",
                "St.",
                "Inc.",
                "Ltd.",
                "Co.",
                "vs.",
                "etc.",
            )
        private val decimalTailRegex = Regex("\\d\\.$")

        private fun hasTrailingKnownAbbreviation(text: String): Boolean =
            knownAbbreviations.any { abbreviation ->
                text.endsWith(" $abbreviation")
            }
    }
}
