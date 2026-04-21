package ragas.testset

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import ragas.testset.graph.NodeType
import ragas.testset.synthesizers.SamplingMode
import ragas.testset.synthesizers.SynthesisControls
import ragas.testset.synthesizers.TestsetGenerator
import ragas.testset.transforms.AdjacentChunkRelationshipBuilder
import ragas.testset.transforms.EmbeddingsTopicExtractor
import ragas.testset.transforms.LlmBasedSummaryExtractor
import ragas.testset.transforms.RegexEntityExtractor
import ragas.testset.transforms.SentenceChunkSplitter
import ragas.testset.transforms.SequenceTransforms
import ragas.testset.transforms.SharedKeywordRelationshipBuilder
import ragas.testset.transforms.SingleTransform
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class WS6ProductionParityTest {
    @Test
    fun ws6FixtureConformanceForSynthesizedOutputStructureAndQuality() =
        runBlocking {
            val fixture = readFixture()
            val documents =
                fixture.jsonObject
                    .getValue("documents")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }
            val config = fixture.jsonObject.getValue("config").jsonObject
            val expected = fixture.jsonObject.getValue("expected").jsonObject

            val testsetSize = config.getValue("testset_size").jsonPrimitive.int
            val maxSentencesPerChunk = config.getValue("max_sentences_per_chunk").jsonPrimitive.int
            val minSharedKeywords = config.getValue("min_shared_keywords").jsonPrimitive.int

            val generator = TestsetGenerator()
            val testset =
                generator.generateFromDocuments(
                    documents = documents,
                    testsetSize = testsetSize,
                    transforms =
                        SequenceTransforms(
                            listOf(
                                SingleTransform(LlmBasedSummaryExtractor()),
                                SingleTransform(RegexEntityExtractor()),
                                SingleTransform(EmbeddingsTopicExtractor()),
                                SingleTransform(SentenceChunkSplitter(maxSentencesPerChunk = maxSentencesPerChunk)),
                                SingleTransform(AdjacentChunkRelationshipBuilder()),
                                SingleTransform(
                                    SharedKeywordRelationshipBuilder(
                                        minSharedKeywords = minSharedKeywords,
                                    ),
                                ),
                            ),
                        ),
                    synthesisControls =
                        SynthesisControls(
                            seed = 17,
                            useRankBiasedSampling = true,
                            multiHopCount = 1,
                        ),
                )

            val graph = generator.knowledgeGraph
            val chunkCount = graph.nodes.count { node -> node.type == NodeType.CHUNK }
            val childRelationshipCount = graph.relationships.count { rel -> rel.type == "child" }
            val nextRelationshipCount = graph.relationships.count { rel -> rel.type == "next" }
            val overlapRelationshipCount = graph.relationships.count { rel -> rel.type == "semantic_overlap" }

            assertTrue(
                chunkCount >= expected.getValue("min_chunk_nodes").jsonPrimitive.int,
                "Expected at least ${expected.getValue("min_chunk_nodes").jsonPrimitive.int} chunk nodes; got $chunkCount",
            )
            assertTrue(
                childRelationshipCount >= expected.getValue("min_child_relationships").jsonPrimitive.int,
                "Expected at least ${expected.getValue(
                    "min_child_relationships",
                ).jsonPrimitive.int} child relationships; got $childRelationshipCount",
            )
            assertTrue(
                nextRelationshipCount >= expected.getValue("min_next_relationships").jsonPrimitive.int,
                "Expected at least ${expected.getValue(
                    "min_next_relationships",
                ).jsonPrimitive.int} next relationships; got $nextRelationshipCount",
            )
            assertTrue(
                overlapRelationshipCount >= expected.getValue("min_overlap_relationships").jsonPrimitive.int,
                "Expected at least ${expected.getValue(
                    "min_overlap_relationships",
                ).jsonPrimitive.int} semantic_overlap relationships; got $overlapRelationshipCount",
            )

            assertEquals(testsetSize, testset.samples.size)

            val synthesizers = testset.samples.map { sample -> sample.synthesizerName }.toSet()
            val requiredSynthesizerPrefixes =
                expected
                    .getValue("required_synthesizer_prefixes")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }
                    .toSet()
            assertTrue(
                requiredSynthesizerPrefixes.all { prefix -> synthesizers.any { name -> name.startsWith(prefix) } },
                "Expected synthesizer families $requiredSynthesizerPrefixes; got $synthesizers",
            )

            val minResponseChars = expected.getValue("min_response_characters").jsonPrimitive.int
            testset.samples.forEach { sample ->
                val evalSample = sample.evalSample as SingleTurnSample
                val question = evalSample.userInput.orEmpty()
                val response = evalSample.response.orEmpty()
                val retrieved = evalSample.retrievedContexts.orEmpty()
                val reference = evalSample.reference.orEmpty()

                assertTrue(question.isNotBlank(), "Generated question should be non-blank.")
                assertTrue(response.length >= minResponseChars, "Generated response should have minimum length.")
                assertTrue(reference.isNotBlank(), "Generated reference should be non-blank.")
                assertTrue(retrieved.isNotEmpty(), "Generated sample should include retrieved contexts.")
                assertTrue(
                    lexicalOverlap(response, retrieved.joinToString(" ")) > 0,
                    "Generated response should share lexical overlap with retrieved contexts.",
                )

                if (sample.synthesizerName.startsWith("multi_hop_")) {
                    assertEquals(2, retrieved.size, "Multi-hop samples should include two retrieved contexts.")
                }
            }
        }

    @Test
    fun seededSamplingProducesDeterministicSynthesizerSelection() =
        runBlocking {
            val documents =
                listOf(
                    "Kotlin evaluation uses stable fixtures for reproducible scoring. Teams compare metric shifts across releases.",
                    "Ragas workflow synthesis combines ranking, chunking, and relationship graphs for test generation.",
                )

            suspend fun generate(seed: Int): List<Pair<String, String>> {
                val generator = TestsetGenerator()
                val testset =
                    generator.generateFromDocuments(
                        documents = documents,
                        testsetSize = 4,
                        transforms =
                            SequenceTransforms(
                                listOf(
                                    SingleTransform(LlmBasedSummaryExtractor()),
                                    SingleTransform(RegexEntityExtractor()),
                                    SingleTransform(EmbeddingsTopicExtractor()),
                                    SingleTransform(SentenceChunkSplitter(maxSentencesPerChunk = 1)),
                                    SingleTransform(AdjacentChunkRelationshipBuilder()),
                                    SingleTransform(SharedKeywordRelationshipBuilder(minSharedKeywords = 1)),
                                ),
                            ),
                        synthesisControls =
                            SynthesisControls(
                                seed = seed,
                                useRankBiasedSampling = true,
                                multiHopCount = 1,
                            ),
                    )
                return testset.samples.map { sample ->
                    val question = (sample.evalSample as SingleTurnSample).userInput.orEmpty()
                    sample.synthesizerName to question
                }
            }

            suspend fun generateTopK(seed: Int): List<Pair<String, String>> {
                val generator = TestsetGenerator()
                val testset =
                    generator.generateFromDocuments(
                        documents = documents,
                        testsetSize = 4,
                        transforms =
                            SequenceTransforms(
                                listOf(
                                    SingleTransform(LlmBasedSummaryExtractor()),
                                    SingleTransform(RegexEntityExtractor()),
                                    SingleTransform(EmbeddingsTopicExtractor()),
                                    SingleTransform(SentenceChunkSplitter(maxSentencesPerChunk = 1)),
                                    SingleTransform(AdjacentChunkRelationshipBuilder()),
                                    SingleTransform(SharedKeywordRelationshipBuilder(minSharedKeywords = 1)),
                                ),
                            ),
                        synthesisControls =
                            SynthesisControls(
                                seed = seed,
                                samplingMode = SamplingMode.TOP_K,
                                multiHopCount = 1,
                            ),
                    )
                return testset.samples.map { sample ->
                    val question = (sample.evalSample as SingleTurnSample).userInput.orEmpty()
                    sample.synthesizerName to question
                }
            }

            val runA = generate(seed = 99)
            val runB = generate(seed = 99)
            val topA = generateTopK(seed = 1)
            val topB = generateTopK(seed = 123)

            assertEquals(runA, runB, "Same seed should produce identical synthesizer/question selection.")
            assertEquals(topA, topB, "Top-K mode should ignore seed and remain deterministic by ranking only.")
        }

    @Test
    fun temperatureSamplingModeIsDeterministicForSameSeed() =
        runBlocking {
            val documents =
                listOf(
                    "Document Alpha explains retrieval scoring and calibration. Alpha chunk two adds evaluator details.",
                    "Document Beta explains synthesis ranking and confidence. Beta chunk two adds orchestration notes.",
                )

            suspend fun generate(seed: Int): List<Pair<String, String>> {
                val generator = TestsetGenerator()
                val testset =
                    generator.generateFromDocuments(
                        documents = documents,
                        testsetSize = 4,
                        transforms =
                            SequenceTransforms(
                                listOf(
                                    SingleTransform(LlmBasedSummaryExtractor()),
                                    SingleTransform(RegexEntityExtractor()),
                                    SingleTransform(EmbeddingsTopicExtractor()),
                                    SingleTransform(SentenceChunkSplitter(maxSentencesPerChunk = 1)),
                                    SingleTransform(AdjacentChunkRelationshipBuilder()),
                                    SingleTransform(SharedKeywordRelationshipBuilder(minSharedKeywords = 1)),
                                ),
                            ),
                        synthesisControls =
                            SynthesisControls(
                                seed = seed,
                                samplingMode = SamplingMode.TEMPERATURE,
                                temperature = 0.65,
                                multiHopCount = 1,
                            ),
                    )
                return testset.samples.map { sample ->
                    val question = (sample.evalSample as SingleTurnSample).userInput.orEmpty()
                    sample.synthesizerName to question
                }
            }

            val runA = generate(seed = 202)
            val runB = generate(seed = 202)
            assertEquals(runA, runB)
        }

    @Test
    fun documentDiversityCapLimitsSingleHopSamplesPerDocument() =
        runBlocking {
            val documents =
                listOf(
                    "AlphaDoc topic one covers retrieval metrics. AlphaDoc topic two covers trace stability.",
                    "BetaDoc topic one covers candidate ranking. BetaDoc topic two covers deterministic controls.",
                )
            val generator = TestsetGenerator()
            val testset =
                generator.generateFromDocuments(
                    documents = documents,
                    testsetSize = 2,
                    transforms =
                        SequenceTransforms(
                            listOf(
                                SingleTransform(LlmBasedSummaryExtractor()),
                                SingleTransform(RegexEntityExtractor()),
                                SingleTransform(EmbeddingsTopicExtractor()),
                                SingleTransform(SentenceChunkSplitter(maxSentencesPerChunk = 1)),
                                SingleTransform(AdjacentChunkRelationshipBuilder()),
                                SingleTransform(SharedKeywordRelationshipBuilder(minSharedKeywords = 1)),
                            ),
                        ),
                    synthesisControls =
                        SynthesisControls(
                            seed = 7,
                            samplingMode = SamplingMode.TOP_K,
                            multiHopCount = 0,
                            singleHopCount = 2,
                            enforceDocumentDiversity = true,
                            maxSingleHopPerDocument = 1,
                        ),
                )

            val documentMarkers =
                testset.samples.mapNotNull { sample ->
                    val context =
                        (sample.evalSample as SingleTurnSample)
                            .retrievedContexts
                            .orEmpty()
                            .firstOrNull()
                            .orEmpty()
                    when {
                        "AlphaDoc" in context -> "AlphaDoc"
                        "BetaDoc" in context -> "BetaDoc"
                        else -> null
                    }
                }

            assertEquals(2, testset.samples.size)
            assertEquals(2, documentMarkers.toSet().size)
        }

    private fun lexicalOverlap(
        left: String,
        right: String,
    ): Int {
        val leftTokens = tokenize(left)
        val rightTokens = tokenize(right)
        return leftTokens.intersect(rightTokens).size
    }

    private fun tokenize(text: String): Set<String> =
        text
            .lowercase()
            .split(Regex("[^a-z0-9]+"))
            .filter { token -> token.length >= 4 }
            .toSet()

    private fun readFixture() =
        Json.parseToJsonElement(
            requireNotNull(javaClass.classLoader.getResourceAsStream("fixtures/testset/ws6_synthesized_output_fixture.json")) {
                "Fixture not found on classpath: fixtures/testset/ws6_synthesized_output_fixture.json"
            }.bufferedReader().use { it.readText() },
        )
}
