package com.youtu.graphrag.shared.prompt

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import com.youtu.graphrag.shared.config.ConfigManager
import com.youtu.graphrag.shared.constructor.KTBuilder
import com.youtu.graphrag.shared.decomposer.GraphQ
import com.youtu.graphrag.shared.graph.GraphNode
import com.youtu.graphrag.shared.graph.GraphRelationship
import com.youtu.graphrag.shared.llm.LlmClient
import com.youtu.graphrag.shared.retriever.IrcotPromptSource
import com.youtu.graphrag.shared.retriever.KTRetriever
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import kotlin.io.path.createDirectories
import kotlin.io.path.pathString
import kotlin.io.path.writeText
import kotlin.test.Test
import kotlin.test.assertEquals

class PromptSnapshotMatrixTest {
    private val mapper = ObjectMapper().registerKotlinModule()

    @Test
    fun `rendered prompt snapshot matrix matches expected dataset mode variants`() {
        val datasets = listOf("demo", "anony_chs", "anony_eng")
        val modes = listOf("noagent", "agent")
        val snapshots = linkedMapOf<String, String>()

        datasets.forEach { dataset ->
            modes.forEach { mode ->
                val root = Files.createTempDirectory("youtu-graphrag-prompt-matrix-$dataset-$mode")
                val env = createFixtureEnv(root = root, dataset = dataset)
                val question = questionForDataset(dataset)
                val query = queryForDataset(dataset)
                val context = contextForDataset(dataset)

                val config = createConfig(env = env, mode = mode)

                val constructionPrompt = renderConstructionPrompt(dataset = dataset, mode = mode, config = config, env = env)
                val decompositionPrompt =
                    renderDecompositionPrompt(dataset = dataset, config = config, schemaPath = env.schemaPath, question = question)
                val retriever = createRetriever(dataset = dataset, mode = mode, config = config, env = env)
                val retrievalPrompt = retriever.generatePrompt(question = question, context = context)
                val ircotBackendPrompt =
                    retriever.generateIrcotPrompt(
                        currentQuery = query,
                        originalQuestion = question,
                        context = context,
                        previousThoughts = "none",
                        step = 2,
                        promptSource = IrcotPromptSource.BACKEND,
                    )
                val ircotMainPrompt =
                    retriever.generateIrcotPrompt(
                        currentQuery = query,
                        originalQuestion = question,
                        context = context,
                        previousThoughts = "none",
                        step = 3,
                        promptSource = IrcotPromptSource.MAIN,
                    )

                snapshots["$dataset.$mode.construction"] = sha256(constructionPrompt)
                snapshots["$dataset.$mode.decomposition"] = sha256(decompositionPrompt)
                snapshots["$dataset.$mode.retrieval"] = sha256(retrievalPrompt)
                snapshots["$dataset.$mode.retrieval.ircot_backend"] = sha256(ircotBackendPrompt)
                snapshots["$dataset.$mode.retrieval.ircot_main"] = sha256(ircotMainPrompt)

                val sourceConfig = createConfig(env = env, mode = mode)
                sourceConfig.overrideConfig(
                    mapOf(
                        "prompts" to
                            mapOf(
                                "retrieval" to
                                    mapOf(
                                        "ircot_backend" to "BACKEND::{original_question}::{current_iteration_query}::{step}",
                                        "ircot_main" to "MAIN::{current_query}::{step}",
                                    ),
                            ),
                    ),
                )
                val sourceRetriever = createRetriever(dataset = dataset, mode = mode, config = sourceConfig, env = env)
                val explicitBackendPrompt =
                    sourceRetriever.generateIrcotPrompt(
                        currentQuery = query,
                        originalQuestion = question,
                        context = context,
                        previousThoughts = "none",
                        step = 4,
                        promptSource = IrcotPromptSource.BACKEND,
                    )
                val explicitMainPrompt =
                    sourceRetriever.generateIrcotPrompt(
                        currentQuery = query,
                        originalQuestion = question,
                        context = context,
                        previousThoughts = "none",
                        step = 5,
                        promptSource = IrcotPromptSource.MAIN,
                    )

                snapshots["$dataset.$mode.retrieval.ircot_backend_explicit"] = sha256(explicitBackendPrompt)
                snapshots["$dataset.$mode.retrieval.ircot_main_explicit"] = sha256(explicitMainPrompt)
            }
        }

        assertEquals(42, snapshots.size)
        assertEquals(
            EXPECTED_HASHES,
            snapshots,
            "Snapshot mismatch. Actual hashes:\n${snapshots.entries.joinToString("\n") { (k, v) -> "\"$k\" to \"$v\"," }}",
        )
    }

    private fun createConfig(
        env: FixtureEnv,
        mode: String,
    ): ConfigManager {
        val config = ConfigManager("config/base_config.json")
        config.overrideConfig(
            mapOf(
                "construction" to mapOf("mode" to mode),
                "triggers" to mapOf("mode" to mode),
                "datasets" to
                    mapOf(
                        env.dataset to
                            mapOf(
                                "corpus_path" to env.corpusPath.pathString,
                                "qa_path" to env.qaPath.pathString,
                                "schema_path" to env.schemaPath.pathString,
                                "graph_output" to env.graphPath.pathString,
                            ),
                    ),
            ),
        )
        return config
    }

    private fun renderConstructionPrompt(
        dataset: String,
        mode: String,
        config: ConfigManager,
        env: FixtureEnv,
    ): String {
        var observedPrompt = ""
        val builder =
            KTBuilder(
                datasetName = dataset,
                schemaPath = env.schemaPath.pathString,
                mode = mode,
                config = config,
                rootDir = env.root,
                llmClient =
                    object : LlmClient {
                        override fun complete(prompt: String): String {
                            observedPrompt = prompt
                            return """
                                {
                                  "attributes": {"PERSON#1": ["role: leader"]},
                                  "triples": [["PERSON#1", "related_to", "ORG#1"]],
                                  "entity_types": {"PERSON#1": "person", "ORG#1": "organization"}
                                }
                                """.trimIndent()
                        }
                    },
            )
        builder.buildKnowledgeGraph(env.corpusPath.pathString)
        return observedPrompt
    }

    private fun renderDecompositionPrompt(
        dataset: String,
        config: ConfigManager,
        schemaPath: Path,
        question: String,
    ): String {
        var observedPrompt = ""
        val graphQ =
            GraphQ(
                datasetName = dataset,
                config = config,
                llmClient =
                    object : LlmClient {
                        override fun complete(prompt: String): String {
                            observedPrompt = prompt
                            return ""
                        }
                    },
            )
        graphQ.decompose(question, schemaPath.pathString)
        return observedPrompt
    }

    private fun createRetriever(
        dataset: String,
        mode: String,
        config: ConfigManager,
        env: FixtureEnv,
    ): KTRetriever =
        KTRetriever.createAndBuild(
            datasetName = dataset,
            graphPath = env.graphPath.pathString,
            recallPaths = 1,
            schemaPath = env.schemaPath.pathString,
            topK = 5,
            mode = mode,
            config = config,
            rootDir = env.root,
        )

    private fun createFixtureEnv(
        root: Path,
        dataset: String,
    ): FixtureEnv {
        val schemaPath = root.resolve("schemas/$dataset.json").also { it.parent.createDirectories() }
        val corpusPath = root.resolve("data/${dataset}_corpus.json").also { it.parent.createDirectories() }
        val qaPath = root.resolve("data/$dataset.json").also { it.parent.createDirectories() }
        val graphPath = root.resolve("output/graphs/${dataset}_new.json").also { it.parent.createDirectories() }
        val chunksPath = root.resolve("output/chunks/$dataset.txt").also { it.parent.createDirectories() }

        schemaPath.writeText("""{"Nodes":["person","organization"],"Relations":["related_to"],"Attributes":["name"]}""")
        mapper.writeValue(
            corpusPath.toFile(),
            listOf(
                mapOf(
                    "title" to "Doc",
                    "text" to chunkForDataset(dataset),
                ),
            ),
        )
        qaPath.writeText("[]")
        mapper.writeValue(
            graphPath.toFile(),
            listOf(
                GraphRelationship(
                    startNode = GraphNode(label = "person", properties = mapOf("name" to "PERSON#1", "chunk id" to "0")),
                    relation = "related_to",
                    endNode = GraphNode(label = "organization", properties = mapOf("name" to "ORG#1", "chunk id" to "0")),
                ),
            ),
        )
        chunksPath.writeText("id:0\tChunk: ${chunkForDataset(dataset)}\n")

        return FixtureEnv(
            root = root,
            dataset = dataset,
            schemaPath = schemaPath,
            corpusPath = corpusPath,
            qaPath = qaPath,
            graphPath = graphPath,
        )
    }

    private fun chunkForDataset(dataset: String): String =
        when (dataset) {
            "anony_chs" -> "PERSON#1在LOCATION#1策划行动。"
            "anony_eng" -> "PERSON#1 planned an operation at LOCATION#1."
            else -> "Alice leads Project Alpha in Tokyo."
        }

    private fun questionForDataset(dataset: String): String =
        when (dataset) {
            "anony_chs" -> "PERSON#1是谁？"
            "anony_eng" -> "Who is PERSON#1?"
            else -> "Who leads Project Alpha?"
        }

    private fun queryForDataset(dataset: String): String =
        when (dataset) {
            "anony_chs" -> "PERSON#1 相关事件"
            "anony_eng" -> "PERSON#1 related events"
            else -> "Project Alpha leader"
        }

    private fun contextForDataset(dataset: String): String =
        when (dataset) {
            "anony_chs" -> "PERSON#1在LOCATION#1策划行动。"
            "anony_eng" -> "PERSON#1 planned an operation at LOCATION#1."
            else -> "Project Alpha is led by Alice."
        }

    private fun sha256(value: String): String =
        MessageDigest
            .getInstance("SHA-256")
            .digest(value.toByteArray(Charsets.UTF_8))
            .joinToString("") { byte -> "%02x".format(byte) }

    private data class FixtureEnv(
        val root: Path,
        val dataset: String,
        val schemaPath: Path,
        val corpusPath: Path,
        val qaPath: Path,
        val graphPath: Path,
    )

    private companion object {
        val EXPECTED_HASHES: Map<String, String> =
            mapOf(
                "demo.noagent.construction" to "b846be1f6f321b91a1295101b4e2d46173fc704fcb8f3cdd678b560320332ebc",
                "demo.noagent.decomposition" to "dc4ba5fe47848329c16dcd3a802c44361903d52be6315b8c2a839d76e0c718d6",
                "demo.noagent.retrieval" to "79d3eb9d158de7070c25da96be7702769d7b710aeffbc53de6717c5628d25602",
                "demo.noagent.retrieval.ircot_backend" to "823f9e081be567b8b5eb9c92f8084794f829344f9c76d043d1326be01398ed83",
                "demo.noagent.retrieval.ircot_main" to "db3fa40797a0c8f834f4372dffdf2c4c0965168324f3340f914a799cf2a71d61",
                "demo.noagent.retrieval.ircot_backend_explicit" to "45bb301c6f32632876641c08aace5611c9a3942ff180b2095b8974c27150a203",
                "demo.noagent.retrieval.ircot_main_explicit" to "fa54c3a5157c89fdd020c2d0eb4a296bb983a0f14ea4e3359767b8389b3a98a6",
                "demo.agent.construction" to "f3c7841ccfb8a406d9fd90b0cdb8ab8dee13f2d054f59bbc3e95902d7f3bb6d1",
                "demo.agent.decomposition" to "dc4ba5fe47848329c16dcd3a802c44361903d52be6315b8c2a839d76e0c718d6",
                "demo.agent.retrieval" to "79d3eb9d158de7070c25da96be7702769d7b710aeffbc53de6717c5628d25602",
                "demo.agent.retrieval.ircot_backend" to "823f9e081be567b8b5eb9c92f8084794f829344f9c76d043d1326be01398ed83",
                "demo.agent.retrieval.ircot_main" to "db3fa40797a0c8f834f4372dffdf2c4c0965168324f3340f914a799cf2a71d61",
                "demo.agent.retrieval.ircot_backend_explicit" to "45bb301c6f32632876641c08aace5611c9a3942ff180b2095b8974c27150a203",
                "demo.agent.retrieval.ircot_main_explicit" to "fa54c3a5157c89fdd020c2d0eb4a296bb983a0f14ea4e3359767b8389b3a98a6",
                "anony_chs.noagent.construction" to "03cccd917b12b61fc7921ff232c2db503186dce0a36136e4a2c0cd36ee7a126a",
                "anony_chs.noagent.decomposition" to "f8e4934009a804f9398e910a0cf6906fbdb8102b27b52e28aa6b04617c7ab30d",
                "anony_chs.noagent.retrieval" to "b98720d6b243d420f4fd649e667d962bb3ad64bb2039c5df378d2accd253fc24",
                "anony_chs.noagent.retrieval.ircot_backend" to "6834d0a3b0ebcaed62c6a88dd0637c220e971153abe7ba33ba05bab5488b79ac",
                "anony_chs.noagent.retrieval.ircot_main" to "a47a2b0a50818357de5a7d85b4a3fc0b67354cafd292deffe09889ae1ca87c2f",
                "anony_chs.noagent.retrieval.ircot_backend_explicit" to "5cff186d3475e538534b6b95231e7cc61f205ce79e781b35b37186550a89d12e",
                "anony_chs.noagent.retrieval.ircot_main_explicit" to "af899725bea35459781a716af8a23e35dd00c45827268f5f93e506085ad5b610",
                "anony_chs.agent.construction" to "655b611db01eccceccf8f73b2a907b2ba46b433f26042386c69e1302272a263f",
                "anony_chs.agent.decomposition" to "f8e4934009a804f9398e910a0cf6906fbdb8102b27b52e28aa6b04617c7ab30d",
                "anony_chs.agent.retrieval" to "b98720d6b243d420f4fd649e667d962bb3ad64bb2039c5df378d2accd253fc24",
                "anony_chs.agent.retrieval.ircot_backend" to "6834d0a3b0ebcaed62c6a88dd0637c220e971153abe7ba33ba05bab5488b79ac",
                "anony_chs.agent.retrieval.ircot_main" to "a47a2b0a50818357de5a7d85b4a3fc0b67354cafd292deffe09889ae1ca87c2f",
                "anony_chs.agent.retrieval.ircot_backend_explicit" to "5cff186d3475e538534b6b95231e7cc61f205ce79e781b35b37186550a89d12e",
                "anony_chs.agent.retrieval.ircot_main_explicit" to "af899725bea35459781a716af8a23e35dd00c45827268f5f93e506085ad5b610",
                "anony_eng.noagent.construction" to "ded63d51163e1221a97214dda6ecec9cad537c63b6aced94c9193c4a61752342",
                "anony_eng.noagent.decomposition" to "a3892963981dce15edbdd7a63943661b0d70edd12b7b7de5dc5c3162c6dd7f06",
                "anony_eng.noagent.retrieval" to "bd5b8a3bbf3f82e8b7f494c79c976ecf1dea6c354cf15d70e8a33156f5de8f8d",
                "anony_eng.noagent.retrieval.ircot_backend" to "253493d6007cc93c2537ecc8181c785f3701aa471439fd4406f6b74e5d5fe0eb",
                "anony_eng.noagent.retrieval.ircot_main" to "62090a31ac46f8184730f2146b167f1a45656d9be463ed152606bd6e55ba45bb",
                "anony_eng.noagent.retrieval.ircot_backend_explicit" to "23cc124316fac1c5098f87f8f449eebda7d4b8ecd9529ac79f4e06154bdf7152",
                "anony_eng.noagent.retrieval.ircot_main_explicit" to "9f85d52f354a98c51205a5b8d324d6f6e494f70a011216ffe5c161552bbf3838",
                "anony_eng.agent.construction" to "00bd9aacdff723b5db7a3499a3af399b247b771382bb21be39f149ee44333874",
                "anony_eng.agent.decomposition" to "a3892963981dce15edbdd7a63943661b0d70edd12b7b7de5dc5c3162c6dd7f06",
                "anony_eng.agent.retrieval" to "bd5b8a3bbf3f82e8b7f494c79c976ecf1dea6c354cf15d70e8a33156f5de8f8d",
                "anony_eng.agent.retrieval.ircot_backend" to "253493d6007cc93c2537ecc8181c785f3701aa471439fd4406f6b74e5d5fe0eb",
                "anony_eng.agent.retrieval.ircot_main" to "62090a31ac46f8184730f2146b167f1a45656d9be463ed152606bd6e55ba45bb",
                "anony_eng.agent.retrieval.ircot_backend_explicit" to "23cc124316fac1c5098f87f8f449eebda7d4b8ecd9529ac79f4e06154bdf7152",
                "anony_eng.agent.retrieval.ircot_main_explicit" to "9f85d52f354a98c51205a5b8d324d6f6e494f70a011216ffe5c161552bbf3838",
            )
    }
}
