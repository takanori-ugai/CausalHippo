package ragas

import kotlinx.coroutines.runBlocking
import ragas.model.MultiTurnSample
import ragas.model.SingleTurnSample
import ragas.testset.graph.KnowledgeGraph
import ragas.testset.graph.Node
import ragas.testset.graph.NodeType
import ragas.testset.graph.Relationship
import ragas.testset.graph.getChildNodes
import ragas.testset.graph.getParentNodes
import ragas.testset.synthesizers.Testset
import ragas.testset.synthesizers.TestsetGenerator
import ragas.testset.synthesizers.TestsetSample
import ragas.testset.transforms.Extractor
import ragas.testset.transforms.Parallel
import ragas.testset.transforms.SequenceTransforms
import ragas.testset.transforms.SingleTransform
import ragas.testset.transforms.applyTransforms
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TestsetScaffoldTest {
    @Test
    fun knowledgeGraphSaveLoadAndQueriesWork() {
        val parent = Node(type = NodeType.DOCUMENT, properties = mutableMapOf("page_content" to "p"))
        val child = Node(type = NodeType.CHUNK, properties = mutableMapOf("page_content" to "c"))
        val graph = KnowledgeGraph(nodes = mutableListOf(parent, child))
        graph.add(Relationship(type = "child", sourceId = parent.id, targetId = child.id))

        val dir = createTempDirectory("ragas-kg").toFile()
        val path = dir.resolve("kg.json").absolutePath
        graph.save(path)

        val loaded = KnowledgeGraph.load(path)
        assertEquals(2, loaded.nodes.size)
        assertEquals(1, loaded.relationships.size)

        val loadedParent = loaded.nodes.first { it.id == parent.id }
        val loadedChild = loaded.nodes.first { it.id == child.id }
        assertEquals(listOf(loadedChild.id), getChildNodes(loadedParent, loaded).map { it.id })
        assertEquals(listOf(loadedParent.id), getParentNodes(loadedChild, loaded).map { it.id })
    }

    @Test
    fun applyTransformsExecutesExtractorPlan() =
        runBlocking {
            val n1 = Node(type = NodeType.DOCUMENT, properties = mutableMapOf("page_content" to "alpha"))
            val n2 = Node(type = NodeType.DOCUMENT, properties = mutableMapOf("page_content" to "beta"))
            val graph = KnowledgeGraph(nodes = mutableListOf(n1, n2))

            val extractor =
                object : Extractor(name = "length_extractor") {
                    override fun propertyName(node: Node): String = "length"

                    override suspend fun extract(node: Node): String {
                        val text = node.getProperty("page_content").orEmpty()
                        return text.length.toString()
                    }
                }

            applyTransforms(
                graph,
                SequenceTransforms(listOf(SingleTransform(extractor))),
            )

            assertEquals("5", n1.getProperty("length"))
            assertEquals("4", n2.getProperty("length"))
        }

    @Test
    fun testsetGeneratorCreatesSingleTurnSamples() =
        runBlocking {
            val generator = TestsetGenerator()
            val testset =
                generator.generateFromDocuments(
                    documents = listOf("Doc one about Kotlin.", "Doc two about RAGAS."),
                    testsetSize = 2,
                )

            assertEquals(2, testset.samples.size)
            val eval = testset.toEvaluationDataset()
            assertEquals(2, eval.samples.size)
            assertTrue(generator.knowledgeGraph.nodes.isNotEmpty())
        }

    @Test
    fun testsetGeneratorPreservesExistingKnowledgeGraphState() =
        runBlocking {
            val existingParent = Node(type = NodeType.DOCUMENT, properties = mutableMapOf("page_content" to "existing"))
            val existingChild = Node(type = NodeType.CHUNK, properties = mutableMapOf("page_content" to "child"))
            val existingRelation = Relationship(type = "child", sourceId = existingParent.id, targetId = existingChild.id)
            val existingGraph =
                KnowledgeGraph(
                    nodes = mutableListOf(existingParent, existingChild),
                    relationships = mutableListOf(existingRelation),
                )
            val generator = TestsetGenerator(existingGraph)

            generator.generateFromDocuments(
                documents = listOf("New document content"),
                testsetSize = 1,
            )

            assertTrue(generator.knowledgeGraph.nodes.any { it.id == existingParent.id })
            assertTrue(generator.knowledgeGraph.nodes.any { it.id == existingChild.id })
            assertTrue(generator.knowledgeGraph.relationships.any { it.id == existingRelation.id })
        }

    @Test
    fun testsetGeneratorSamplesFromCurrentDocumentsNotEntireGraph() =
        runBlocking {
            val existing = Node(type = NodeType.DOCUMENT, properties = mutableMapOf("page_content" to "existing graph doc"))
            val generator = TestsetGenerator(KnowledgeGraph(nodes = mutableListOf(existing)))

            val testset =
                generator.generateFromDocuments(
                    documents = listOf("new document content"),
                    testsetSize = 1,
                )

            val generated = testset.samples.first().evalSample as SingleTurnSample
            val retrieved = generated.retrievedContexts.orEmpty()
            assertEquals(listOf("new document content"), retrieved)
            assertFalse(retrieved.contains("existing graph doc"))
        }

    @Test
    fun testsetToEvaluationDatasetRejectsMixedSampleTypes() {
        val mixed =
            Testset(
                samples =
                    listOf(
                        TestsetSample(
                            evalSample = SingleTurnSample(userInput = "u", response = "r"),
                            synthesizerName = "s1",
                        ),
                        TestsetSample(
                            evalSample = MultiTurnSample(userInput = emptyList()),
                            synthesizerName = "s2",
                        ),
                    ),
            )

        val error = assertFailsWith<IllegalArgumentException> { mixed.toEvaluationDataset() }
        assertTrue(error.message.orEmpty().contains("Mixed sample types are not supported"))
    }

    @Test
    fun testsetFromListFailsFastForMultiTurnRows() {
        val row =
            mapOf<String, Any?>(
                "user_input" to listOf(mapOf("type" to "human", "content" to "Hello")),
                "reference" to "ref",
                "synthesizer_name" to "synth",
            )

        val error =
            assertFailsWith<IllegalArgumentException> {
                Testset.fromList(listOf(row))
            }
        assertTrue(error.message.orEmpty().contains("Multi-turn deserialization is not implemented"))
    }

    @Test
    fun parallelExtractorsDoNotRaceWhenSettingSameProperty() =
        runBlocking {
            val node = Node(type = NodeType.DOCUMENT, properties = mutableMapOf("page_content" to "alpha"))
            val graph = KnowledgeGraph(nodes = mutableListOf(node))

            val extractorA =
                object : Extractor(name = "e1") {
                    override fun propertyName(node: Node): String = "shared_key"

                    override suspend fun extract(node: Node): String = "a"
                }
            val extractorB =
                object : Extractor(name = "e2") {
                    override fun propertyName(node: Node): String = "shared_key"

                    override suspend fun extract(node: Node): String = "b"
                }

            applyTransforms(
                graph,
                Parallel(listOf(extractorA, extractorB)),
            )

            assertTrue(node.getProperty("shared_key") in setOf("a", "b"))
        }

    @Test
    fun nodeConstructorNormalizesPropertyKeysToLowercase() {
        val node = Node(type = NodeType.DOCUMENT, properties = mutableMapOf("Page_Content" to "alpha"))
        assertEquals("alpha", node.getProperty("page_content"))
        assertEquals("alpha", node.getProperty("PAGE_CONTENT"))
    }

    @Test
    fun nodeConstructorRejectsDuplicateKeysAfterLowercaseNormalization() {
        val error =
            assertFailsWith<IllegalArgumentException> {
                Node(
                    type = NodeType.DOCUMENT,
                    properties = mutableMapOf("Page_Content" to "alpha", "page_content" to "beta"),
                )
            }
        assertTrue(error.message.orEmpty().contains("Duplicate property keys when normalized to lowercase"))
    }
}
