package ragas.testset

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.double
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import ragas.model.SingleTurnSample
import kotlin.test.Test
import kotlin.test.assertTrue

class WS6CrossLanguageGoldenTest {
    @Test
    fun ws6CrossLanguageFixtureCalibratesShapePromptStyleAndGraphStats() =
        runBlocking {
            val fixture = WS6TestFixtures.readFixture("ws6_cross_language_golden_fixture.json")
            val root = fixture.jsonObject
            val run = WS6TestFixtures.executeFixture(fixture)
            val pythonReference = root.getValue("python_reference").jsonObject
            val graphReference = root.getValue("graph_reference").jsonObject

            val requiredFields =
                pythonReference
                    .getValue("sample_shape_required_fields")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }
                    .toSet()
            val allowedStyles =
                pythonReference
                    .getValue("allowed_query_style_names")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }
                    .toSet()
            val allowedLengths =
                pythonReference
                    .getValue("allowed_query_length_names")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }
                    .toSet()
            val singleHopMarkers =
                pythonReference
                    .getValue("single_hop_prompt_markers")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }
            val multiHopMarkers =
                pythonReference
                    .getValue("multi_hop_prompt_markers")
                    .jsonArray
                    .map { item -> item.jsonPrimitive.content }

            run.testset.samples.forEach { sample ->
                val row = sample.evalSample.toMap().keys
                assertTrue(requiredFields.all { field -> field in row }, "Missing required shape fields in sample map.")

                val eval = sample.evalSample as SingleTurnSample
                val question = eval.userInput.orEmpty()
                val style = eval.queryStyle.orEmpty()
                val length = eval.queryLength.orEmpty()
                val persona = eval.personaName.orEmpty()
                assertTrue(style in allowedStyles, "query_style '$style' not in Python-style enum names.")
                assertTrue(length in allowedLengths, "query_length '$length' not in Python-style enum names.")
                assertTrue(persona.isNotBlank(), "persona_name should be non-blank for Python-shape parity.")

                if (sample.synthesizerName.startsWith("single_hop_")) {
                    assertTrue(singleHopMarkers.any { marker -> question.contains(marker) })
                }
                if (sample.synthesizerName.startsWith("multi_hop_")) {
                    assertTrue(multiHopMarkers.any { marker -> question.contains(marker) })
                    assertTrue((eval.referenceContexts?.size ?: 0) >= 2, "multi-hop should include multiple contexts.")
                }
            }

            val totalPairs = run.chunkCount * (run.chunkCount - 1) / 2
            val overlapDensity =
                if (totalPairs == 0) {
                    0.0
                } else {
                    run.overlapRelationshipCount.toDouble() / totalPairs.toDouble()
                }

            assertTrue(run.chunkCount >= graphReference.getValue("chunk_count_min").jsonPrimitive.int)
            assertTrue(run.chunkCount <= graphReference.getValue("chunk_count_max").jsonPrimitive.int)
            assertTrue(run.childRelationshipCount >= graphReference.getValue("child_edges_min").jsonPrimitive.int)
            assertTrue(run.nextRelationshipCount >= graphReference.getValue("next_edges_min").jsonPrimitive.int)
            assertTrue(overlapDensity >= graphReference.getValue("overlap_density_min").jsonPrimitive.double)
            assertTrue(overlapDensity <= graphReference.getValue("overlap_density_max").jsonPrimitive.double)
        }
}
