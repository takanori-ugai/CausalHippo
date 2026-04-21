package ragas.testset

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.testset.graph.KnowledgeGraph
import ragas.testset.graph.NodeType
import ragas.testset.synthesizers.SamplingMode
import ragas.testset.synthesizers.SynthesisControls
import ragas.testset.synthesizers.Testset
import ragas.testset.synthesizers.TestsetGenerator
import ragas.testset.transforms.AdjacentChunkRelationshipBuilder
import ragas.testset.transforms.EmbeddingsTopicExtractor
import ragas.testset.transforms.LlmBasedSummaryExtractor
import ragas.testset.transforms.RegexEntityExtractor
import ragas.testset.transforms.SentenceChunkSplitter
import ragas.testset.transforms.SequenceTransforms
import ragas.testset.transforms.SharedKeywordRelationshipBuilder
import ragas.testset.transforms.SingleTransform

internal object WS6TestFixtures {
    data class FixtureRun(
        val testset: Testset,
        val graph: KnowledgeGraph,
        val chunkCount: Int,
        val childRelationshipCount: Int,
        val nextRelationshipCount: Int,
        val overlapRelationshipCount: Int,
    )

    fun readFixture(name: String): JsonElement =
        Json.parseToJsonElement(
            requireNotNull(javaClass.classLoader.getResourceAsStream("fixtures/testset/$name")) {
                "Fixture not found on classpath: fixtures/testset/$name"
            }.bufferedReader().use { it.readText() },
        )

    suspend fun executeFixture(fixture: JsonElement): FixtureRun {
        val root = fixture.jsonObject
        val documents = root.getValue("documents").jsonArray.map { item -> item.jsonPrimitive.content }
        val config = root.getValue("config").jsonObject

        val testsetSize = config.getValue("testset_size").jsonPrimitive.int
        val maxSentencesPerChunk = config.getValue("max_sentences_per_chunk").jsonPrimitive.int
        val minSharedKeywords = config.getValue("min_shared_keywords").jsonPrimitive.int
        val controls = synthesisControlsFrom(config.getValue("synthesis_controls").jsonObject)

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
                synthesisControls = controls,
            )

        val graph = generator.knowledgeGraph
        return FixtureRun(
            testset = testset,
            graph = graph,
            chunkCount = graph.nodes.count { node -> node.type == NodeType.CHUNK },
            childRelationshipCount = graph.relationships.count { rel -> rel.type == "child" },
            nextRelationshipCount = graph.relationships.count { rel -> rel.type == "next" },
            overlapRelationshipCount = graph.relationships.count { rel -> rel.type == "semantic_overlap" },
        )
    }

    private fun synthesisControlsFrom(obj: JsonObject): SynthesisControls {
        val modeString = obj["sampling_mode"]?.jsonPrimitive?.content
        val mode =
            when (modeString) {
                "top_k" -> SamplingMode.TOP_K
                "rank_biased" -> SamplingMode.RANK_BIASED
                "temperature" -> SamplingMode.TEMPERATURE
                null -> null
                else -> error("Unknown sampling_mode in fixture: $modeString")
            }

        return SynthesisControls(
            seed = obj["seed"]?.jsonPrimitive?.int ?: 42,
            samplingMode = mode,
            temperature = obj["temperature"]?.jsonPrimitive?.double ?: 0.8,
            singleHopCount = obj["single_hop_count"]?.jsonPrimitive?.int,
            multiHopCount = obj["multi_hop_count"]?.jsonPrimitive?.int,
            enforceDocumentDiversity = obj["enforce_document_diversity"]?.jsonPrimitive?.boolean ?: true,
            maxSingleHopPerDocument = obj["max_single_hop_per_document"]?.jsonPrimitive?.int ?: 2,
            useRankBiasedSampling = obj["use_rank_biased_sampling"]?.jsonPrimitive?.boolean ?: true,
        )
    }
}
