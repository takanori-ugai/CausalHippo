package lightrag.operate

import dev.langchain4j.data.embedding.Embedding
import dev.langchain4j.store.embedding.CosineSimilarity
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.CancellationException
import lightrag.core.Constants
import lightrag.core.types.BaseVectorStorage
import kotlin.math.round

/** Select chunks only from provenance attached to retained graph hits. */
internal class RelatedChunkSelector(
    query: String,
    private val vectors: BaseVectorStorage?,
    config: Map<String, Any?>,
) {
    private val perGroup = (config["related_chunk_number"] as? Number)?.toInt() ?: 5
    private val useVectors = (config["kg_chunk_pick_method"] ?: "VECTOR").toString().equals("VECTOR", ignoreCase = true)
    private val queryVector by lazy { vectors?.embeddingFunc?.embed(query)?.content() }

    suspend fun select(
        rows: List<Map<String, Any>>,
        exclude: Set<String> = emptySet(),
    ): List<String> {
        if (perGroup <= 0) return emptyList()
        val groups =
            rows.map { row ->
                row["source_id"]
                    ?.toString()
                    .orEmpty()
                    .split(Constants.GRAPH_FIELD_SEP)
                    .map(String::trim)
                    .filter { it.isNotEmpty() && it !in exclude }
                    .distinct()
            }
        val counts = groups.flatten().groupingBy { it }.eachCount()
        val seen = mutableSetOf<String>()
        val ranked = groups.map { ids -> ids.filter(seen::add).sortedByDescending(counts::getValue) }.filter(List<String>::isNotEmpty)
        if (ranked.isEmpty()) return emptyList()
        if (vectors != null && useVectors) {
            val ids = ranked.flatten()
            try {
                val stored = vectors.getVectorsByIds(ids)
                if (ids.all(stored::containsKey)) {
                    val embedding = queryVector
                    if (embedding != null && ids.all { stored.getValue(it).size == embedding.dimension() }) {
                        val scores =
                            ids.map { id ->
                                id to
                                    CosineSimilarity.between(embedding, Embedding(stored.getValue(id).toFloatArray()))
                            }
                        if (scores.all { it.second.isFinite() }) {
                            val quota = (perGroup.toLong() * ranked.size / 2).coerceIn(1, ids.size.toLong()).toInt()
                            return scores.sortedByDescending { it.second }.take(quota).map { it.first }
                        }
                    }
                }
            } catch (ex: CancellationException) {
                throw ex
            } catch (ex: RuntimeException) {
                logger.warn(ex) { "Could not rank LightRAG source chunks by vector similarity; using graph support." }
            }
        }
        val selected = mutableListOf<String>()
        val leftovers = mutableListOf<String>()
        var unused = 0L
        for ((index, ids) in ranked.withIndex()) {
            val quota = if (ranked.size == 1) perGroup else round(perGroup - (perGroup - 1.0) * index / (ranked.size - 1)).toInt()
            selected += ids.take(quota)
            leftovers += ids.drop(quota)
            unused += (quota - ids.size).coerceAtLeast(0)
        }
        return selected + leftovers.take(minOf(unused, leftovers.size.toLong()).toInt())
    }

    private companion object {
        val logger = KotlinLogging.logger {}
    }
}
