package shared.rag

import io.mockk.coEvery
import io.mockk.mockkStatic
import io.mockk.unmockkAll
import kotlinx.coroutines.runBlocking
import pathrag.PathRAG
import pathrag.base.QueryParam
import java.io.File
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class CommonRagTest {
    @AfterTest
    fun cleanup() {
        unmockkAll()
        File("build/tmp")
            .listFiles { file -> file.isDirectory && file.name.startsWith("common_rag_test_") }
            ?.forEach { dir -> dir.deleteRecursively() }
    }

    @Test
    fun `sync common rag operations work through pathrag`() =
        runBlocking {
            mockPathRagLlm()
            val workingDir = "build/tmp/common_rag_test_sync_${System.currentTimeMillis()}"

            PathRAG(
                workingDir = workingDir,
                chunkTokenSize = 32,
                chunkOverlapTokenSize = 8,
            ).use { pathRag ->
                val rag: CommonRag<QueryParam, String> = pathRag

                rag.upsert("Alpha and Beta are connected in one document.")
                assertTrue(pathRag.graph().hasNode("ALPHA"))
                assertTrue(pathRag.graph().hasEdge("ALPHA", "BETA"))

                val context = rag.query("How are Alpha and Beta related?", QueryParam(mode = "local", onlyNeedContext = true, topK = 5))
                assertTrue(context.contains("local-information"))

                val snapshotPath = "$workingDir/graph_snapshot_sync.json"
                rag.saveGraph(snapshotPath)
                assertTrue(File(snapshotPath).exists())

                rag.drop()
                assertFalse(pathRag.graph().hasNode("ALPHA"))

                rag.loadGraph(snapshotPath)
                assertTrue(pathRag.graph().hasNode("ALPHA"))
                assertTrue(pathRag.graph().hasEdge("ALPHA", "BETA"))
            }
        }

    @Test
    fun `async common rag operations work through pathrag`() =
        runBlocking {
            mockPathRagLlm()
            val workingDir = "build/tmp/common_rag_test_async_${System.currentTimeMillis()}"

            PathRAG(
                workingDir = workingDir,
                chunkTokenSize = 32,
                chunkOverlapTokenSize = 8,
            ).use { pathRag ->
                val rag: CommonRag<QueryParam, String> = pathRag

                rag.aupsert(listOf("Alpha and Beta are connected in another document."))
                assertTrue(pathRag.graph().hasNode("ALPHA"))
                assertTrue(pathRag.graph().hasEdge("ALPHA", "BETA"))

                val context = rag.aquery("Tell me about Alpha and Beta", QueryParam(mode = "local", onlyNeedContext = true, topK = 5))
                assertTrue(context.contains("local-information"))

                val snapshotPath = "$workingDir/graph_snapshot_async.json"
                rag.asaveGraph(snapshotPath)
                assertTrue(File(snapshotPath).exists())

                rag.adrop()
                assertFalse(pathRag.graph().hasNode("ALPHA"))

                rag.aloadGraph(snapshotPath)
                assertTrue(pathRag.graph().hasNode("ALPHA"))
                assertTrue(pathRag.graph().hasEdge("ALPHA", "BETA"))
            }
        }

    private fun mockPathRagLlm() {
        mockkStatic("pathrag.llm.LlmKt")
        coEvery {
            pathrag.llm.openAiComplete(
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
                any(),
            )
        } answers { call ->
            val prompt = call.invocation.args[1] as String
            val keywordExtraction = call.invocation.args[4] as Boolean
            when {
                keywordExtraction -> {
                    """{"high_level_keywords":["alpha"],"low_level_keywords":["beta"]}"""
                }

                prompt.contains("Entity_types:") -> {
                    """
                    {
                      "entities": [
                        { "entity_name": "ALPHA", "entity_type": "THING", "description": "alpha desc" },
                        { "entity_name": "BETA", "entity_type": "THING", "description": "beta desc" }
                      ],
                      "relationships": [
                        { "src_id": "ALPHA", "tgt_id": "BETA", "description": "connects", "keywords": "k", "weight": 1.0 }
                      ]
                    }
                    """.trimIndent()
                }

                else -> {
                    "ANSWER"
                }
            }
        }
        coEvery { pathrag.llm.openAiEmbedding(any(), any()) } answers { call ->
            val inputs = call.invocation.args[0] as List<String>
            inputs.map { DoubleArray(1536) { 0.01 } }
        }
    }
}
