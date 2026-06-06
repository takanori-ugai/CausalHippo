package com.youtu.graphrag.shared.retriever

import org.apache.lucene.analysis.core.KeywordAnalyzer
import org.apache.lucene.document.Document
import org.apache.lucene.document.KnnFloatVectorField
import org.apache.lucene.document.StoredField
import org.apache.lucene.index.DirectoryReader
import org.apache.lucene.index.IndexWriter
import org.apache.lucene.index.IndexWriterConfig
import org.apache.lucene.search.IndexSearcher
import org.apache.lucene.search.KnnFloatVectorQuery
import org.apache.lucene.store.ByteBuffersDirectory
import org.apache.lucene.store.Directory

/**
 * Lucene-backed ANN index (HNSW under the hood) for approximate kNN vector search.
 */
class LuceneAnnIndex<T> private constructor(
    private val directory: Directory,
    private val reader: DirectoryReader,
    private val searcher: IndexSearcher,
    private val items: List<T>,
    private val embedder: TextEmbedder,
) : SemanticVectorIndex<T> {
    override val strategyTag: String = "hybrid_lexical_lucene_ann"

    override fun search(
        query: String,
        limit: Int,
    ): List<VectorSearchHit<T>> {
        if (items.isEmpty() || limit <= 0) {
            return emptyList()
        }

        val queryVector = embedder.embed(query)
        val knnQuery = KnnFloatVectorQuery(FIELD_VECTOR, queryVector, limit)
        val topDocs = searcher.search(knnQuery, limit)
        val storedFields = reader.storedFields()
        return topDocs.scoreDocs.mapNotNull { scoreDoc ->
            val doc = storedFields.document(scoreDoc.doc)
            val ordinal = doc.getField(FIELD_ORDINAL)?.numericValue()?.toInt() ?: return@mapNotNull null
            val item = items.getOrNull(ordinal) ?: return@mapNotNull null
            VectorSearchHit(
                item = item,
                score = scoreDoc.score.toDouble(),
            )
        }
    }

    companion object {
        private const val FIELD_VECTOR = "vector"
        private const val FIELD_ORDINAL = "ordinal"

        fun <T> build(
            items: List<T>,
            embedder: TextEmbedder = HashTextEmbedder(),
            textSelector: (T) -> String,
            vectorSelector: ((T) -> FloatArray)? = null,
        ): LuceneAnnIndex<T> {
            val directory = ByteBuffersDirectory()
            val writerConfig = IndexWriterConfig(KeywordAnalyzer())

            IndexWriter(directory, writerConfig).use { writer ->
                items.forEachIndexed { ordinal, item ->
                    val vector = vectorSelector?.invoke(item) ?: embedder.embed(textSelector(item))
                    val document =
                        Document().apply {
                            add(KnnFloatVectorField(FIELD_VECTOR, vector))
                            add(StoredField(FIELD_ORDINAL, ordinal))
                        }
                    writer.addDocument(document)
                }
            }

            val reader = DirectoryReader.open(directory)
            val searcher = IndexSearcher(reader)
            return LuceneAnnIndex(
                directory = directory,
                reader = reader,
                searcher = searcher,
                items = items,
                embedder = embedder,
            )
        }
    }
}
