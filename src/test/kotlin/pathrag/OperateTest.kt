package pathrag

import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import pathrag.base.BaseGraphStorage
import pathrag.base.BaseKVStorage
import pathrag.base.BaseVectorStorage
import pathrag.base.QueryParam
import pathrag.operate.chunkingByTokenSize
import pathrag.operate.extractEntities
import pathrag.operate.kgQuery
import pathrag.utils.ResponseCache
import pathrag.utils.Tokenizer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

private class FakeGraphStorage : BaseGraphStorage("test", emptyMap()) {
    private val nodes = mutableMapOf<String, MutableMap<String, Any?>>()
    private val edges = mutableMapOf<Pair<String, String>, MutableMap<String, Any?>>()
    private val lock = Mutex()

    override suspend fun hasNode(nodeId: String): Boolean = lock.withLock { nodes.containsKey(nodeId) }

    override suspend fun hasEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Boolean = lock.withLock { edges.containsKey(sourceNodeId to targetNodeId) }

    override suspend fun nodeDegree(nodeId: String): Int = lock.withLock { edges.keys.count { it.first == nodeId || it.second == nodeId } }

    override suspend fun getNode(nodeId: String): Map<String, Any?>? = lock.withLock { nodes[nodeId]?.toMap() }

    override suspend fun getEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ): Map<String, Any?>? = lock.withLock { edges[sourceNodeId to targetNodeId]?.toMap() }

    override suspend fun getNodeEdges(sourceNodeId: String): List<Pair<String, String>> =
        lock.withLock { edges.keys.filter { it.first == sourceNodeId || it.second == sourceNodeId } }

    override suspend fun upsertNode(
        nodeId: String,
        nodeData: Map<String, Any?>,
    ) {
        lock.withLock {
            nodes[nodeId] = (nodes[nodeId] ?: mutableMapOf()).apply { putAll(nodeData) }
        }
    }

    override suspend fun upsertEdge(
        sourceNodeId: String,
        targetNodeId: String,
        edgeData: Map<String, Any?>,
    ) {
        lock.withLock {
            edges[sourceNodeId to targetNodeId] = (edges[sourceNodeId to targetNodeId] ?: mutableMapOf()).apply { putAll(edgeData) }
        }
    }

    override suspend fun deleteEdge(
        sourceNodeId: String,
        targetNodeId: String,
    ) {
        lock.withLock { edges.remove(sourceNodeId to targetNodeId) }
    }

    override suspend fun deleteNode(nodeId: String) {
        lock.withLock {
            nodes.remove(nodeId)
            edges.keys.filter { it.first == nodeId || it.second == nodeId }.forEach { edges.remove(it) }
        }
    }

    override suspend fun nodes(): List<String> = lock.withLock { nodes.keys.toList() }

    override suspend fun edges(): List<Pair<String, String>> = lock.withLock { edges.keys.toList() }
}

private class FakeVectorStorage(
    override val namespace: String,
    private val results: List<Map<String, Any?>>,
    private val writeError: IllegalStateException? = null,
) : BaseVectorStorage(namespace, emptyMap()) {
    override suspend fun query(
        query: String,
        topK: Int,
    ): List<Map<String, Any?>> = results.take(topK)

    override suspend fun upsert(data: Map<String, Map<String, Any?>>) {
        writeError?.let { throw it }
    }
}

private class FakeKVStorage(
    override val namespace: String,
    private val records: Map<String, Map<String, Any>?>,
) : BaseKVStorage<Map<String, Any>>(namespace, emptyMap()) {
    override suspend fun allKeys(): List<String> = records.keys.toList()

    override suspend fun getById(id: String): Map<String, Any>? = records[id]

    override suspend fun getByIds(
        ids: List<String>,
        fields: Set<String>?,
    ): List<Map<String, Any>?> = ids.map { records[it] }

    override suspend fun filterKeys(data: List<String>): Set<String> = data.filterNot { records.containsKey(it) }.toSet()

    override suspend fun upsert(data: Map<String, Map<String, Any>>): Unit = Unit

    override suspend fun drop(): Unit = Unit
}

class OperateTest {
    @Test
    fun chunkingByTokenSize_splitsContent() {
        val text = "Kotlin ".repeat(300)
        val chunks = chunkingByTokenSize(text, overlapTokenSize = 10, maxTokenSize = 50, tiktokenModel = "gpt-4o-mini")
        assertTrue(chunks.size > 1)
        val indices = chunks.map { it["chunk_order_index"] as Int }
        assertEquals(indices.sorted(), indices)
        assertTrue(chunks.all { (it["tokens"] as Int) <= 50 })
    }

    @Test
    fun kgQuery_returnsLocalContext() =
        runBlocking {
            val graph = FakeGraphStorage()
            graph.upsertNode("NODE", mapOf("entity_type" to "THING", "description" to "desc", "source_id" to "doc1"))
            val entitiesVdb =
                FakeVectorStorage(
                    "entities",
                    listOf(mapOf("entity_name" to "NODE", "content" to "desc", "full_doc_id" to "doc1")),
                )
            val relationshipsVdb = FakeVectorStorage("rels", emptyList())
            val kv = FakeKVStorage("chunks", mapOf("doc1" to mapOf("content" to "Document content")))

            val context =
                kgQuery(
                    query = "question",
                    knowledgeGraphInst = graph,
                    entitiesVdb = entitiesVdb,
                    relationshipsVdb = relationshipsVdb,
                    textChunksDb = kv,
                    queryParam = QueryParam(mode = "local", onlyNeedContext = true, topK = 5),
                    globalConfig = emptyMap(),
                    llmModel = { prompt, system, _, _, _, _, _ -> "llm-$prompt-$system" },
                )
            assertTrue(context.contains("local-information"))
            assertTrue(context.contains("NODE"))
            assertTrue(context.contains("Document content"))
        }

    @Test
    fun kgQuery_preservesSourcesQuestionAndPromptOnlyAcrossModes() {
        runBlocking {
            val graph = FakeGraphStorage()
            graph.upsertNode("ALPHA", mapOf("source_id" to "chunk-a<SEP>chunk-b"))
            graph.upsertEdge("ALPHA", "BETA", mapOf("source_id" to "chunk-a<SEP>chunk-b"))
            val entities = FakeVectorStorage("entities", listOf(mapOf("entity_name" to "ALPHA")))
            val relations = FakeVectorStorage("rels", listOf(mapOf("src_id" to "ALPHA", "tgt_id" to "BETA")))
            val kv =
                FakeKVStorage(
                    "chunks",
                    mapOf(
                        "chunk-a" to mapOf("content" to "first source"),
                        "chunk-b" to mapOf("content" to "second source"),
                    ),
                )
            val question = "Compare the two events chronologically."
            val generated = mutableListOf<String>()

            suspend fun run(param: QueryParam): String =
                kgQuery(question, graph, entities, relations, kv, param, emptyMap(), { prompt, system, _, _, _, _, _ ->
                    if (system == null) {
                        """{"low_level_keywords":["ALPHA"],"high_level_keywords":["comparison"]}"""
                    } else {
                        generated.add(prompt)
                        "generated"
                    }
                })

            for (mode in listOf("local", "global", "hybrid")) {
                val prompt = run(QueryParam(mode = mode, onlyNeedPrompt = true))
                assertTrue(prompt.contains("first source"), mode)
                assertTrue(prompt.contains("second source"), mode)
                assertTrue(generated.isEmpty(), mode)
                assertEquals("generated", run(QueryParam(mode = mode)))
                assertEquals(question, generated.removeAt(0), mode)
            }
        }
    }

    @Test
    fun kgQuery_hybridBoundsCompleteContextAndInvalidatesAnswerWhenItChanges() {
        runBlocking {
            val graph = FakeGraphStorage()
            graph.upsertNode("NODE", mapOf("description" to "initial", "source_id" to " "))
            graph.upsertEdge("NODE", "END", mapOf("description" to "linked", "source_id" to "chunk"))
            val entities = FakeVectorStorage("entities", listOf(mapOf("entity_name" to "NODE", "source_id" to "chunk")))
            val relations = FakeVectorStorage("rels", listOf(mapOf("src_id" to "NODE", "tgt_id" to "END")))
            val kv = FakeKVStorage("chunks", mapOf("chunk" to mapOf("content" to "word ".repeat(250))))
            val cache = ResponseCache()
            var answers = 0

            suspend fun run(param: QueryParam): String =
                kgQuery("question", graph, entities, relations, kv, param, emptyMap(), { _, system, _, _, _, _, _ ->
                    if (system == null) {
                        """{"low_level_keywords":["NODE"],"high_level_keywords":["linked"]}"""
                    } else {
                        answers += 1
                        "answer $answers"
                    }
                }, cache)

            val options = QueryParam(mode = "hybrid", topK = 5)
            val evidence = "word ".repeat(250)
            val contextOptions = options.copy(onlyNeedContext = true)
            val fullContext = run(contextOptions)
            assertEquals(1, fullContext.split(evidence).size - 1)
            val budget = Tokenizer.count(fullContext) - 1
            val trimmedContext = run(contextOptions.copy(maxTokenForTextUnit = budget))
            assertTrue(evidence in trimmedContext)
            assertTrue(Tokenizer.count(trimmedContext) <= budget)
            assertTrue(evidence in run(options.copy(mode = "local", onlyNeedContext = true)))
            val first = run(options)
            assertEquals(first, run(options))
            assertEquals(1, answers)
            graph.deleteEdge("NODE", "END")
            assertFalse(first == run(options))
            assertEquals(2, answers)
            assertTrue(Tokenizer.count(run(options.copy(onlyNeedContext = true, maxTokenForTextUnit = 100))) <= 100)
        }
    }

    @Test
    fun kgQuery_retainsDistinctRoutesThroughTheSameNodes() {
        runBlocking {
            val graph = FakeGraphStorage()
            for (node in listOf("A", "B", "C", "D")) graph.upsertNode(node, emptyMap())
            for ((from, to) in listOf("A" to "B", "A" to "C", "B" to "C", "B" to "D", "C" to "D")) {
                graph.upsertEdge(from, to, mapOf("description" to "$from-$to"))
            }
            val context =
                kgQuery(
                    "question",
                    graph,
                    FakeVectorStorage("entities", listOf(mapOf("entity_name" to "A"), mapOf("entity_name" to "D"))),
                    FakeVectorStorage("relations", emptyList()),
                    FakeKVStorage("chunks", emptyMap()),
                    QueryParam(mode = "local", onlyNeedContext = true),
                    mapOf("fixed_low_level_keywords" to listOf("A", "D")),
                    { _, _, _, _, _, _, _ -> error("unexpected model call") },
                )
            val threeEdgeRoutes = context.lineSequence().filter { it.contains("edge(B-C)") }.toList()
            assertEquals(2, threeEdgeRoutes.size)
            assertTrue(threeEdgeRoutes.any { it.contains("edge(A-B)") && it.contains("edge(C-D)") })
            assertTrue(threeEdgeRoutes.any { it.contains("edge(A-C)") && it.contains("edge(B-D)") })
        }
    }

    @Test
    fun extractEntities_propagatesVectorWriteFailureAndRetryKeepsEdgeWeight() {
        runBlocking {
            val graph = FakeGraphStorage()
            val llm: suspend (String, String?, List<Map<String, String>>, Boolean, Boolean, Int?, Any?) -> String =
                { _, _, _, _, _, _, _ ->
                    """{"entities":[{"entity_name":"ALPHA","entity_type":"THING","description":"A"}],
                        "relationships":[{"src_id":"ALPHA","tgt_id":"BETA","description":"link","keywords":"k","weight":1.0}]}"""
                }
            val chunks = mapOf("chunk" to mapOf("content" to "Alpha and Beta"))
            val config = mapOf("llm_model_func" to llm)
            val entities = FakeVectorStorage("entities", emptyList())
            assertFailsWith<IllegalStateException> {
                extractEntities(
                    chunks,
                    graph,
                    entities,
                    FakeVectorStorage("relations", emptyList(), IllegalStateException("write failed")),
                    config,
                )
            }
            extractEntities(chunks, graph, entities, FakeVectorStorage("relations", emptyList()), config)
            assertEquals(1.0, graph.getEdge("ALPHA", "BETA")?.get("weight"))
        }
    }
}
