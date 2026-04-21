package ragas.examples.rageval

import ragas.backends.rowToJsonLine
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.absoluteValue

/**
 * Sample document corpus used by the RAG pipeline example.
 */
val DOCUMENTS: List<String> =
    listOf(
        "Ragas are melodic frameworks in Indian classical music.",
        "There are many types of ragas, each with its own mood and time of day.",
        "Ragas are used to evoke specific emotions in the listener.",
        "The performance of a raga involves improvisation within a set structure.",
        "Ragas can be performed on various instruments or sung vocally.",
    )

/**
 * Captures a single trace event emitted during RAG pipeline execution.
 *
 * @property eventType Trace event type (for example, `init`, `query_start`, or `error`).
 * @property component Component name that emitted the event.
 * @property data Event payload map containing contextual details.
 */
data class TraceEvent(
    val eventType: String,
    val component: String,
    val data: Map<String, Any?>,
)

/**
 * One retrieved document candidate with its relevance metadata.
 *
 * @property content Retrieved document content text.
 * @property similarityScore Keyword-overlap similarity score.
 * @property documentId Source document identifier in the in-memory corpus.
 */
data class RetrievedDoc(
    val content: String,
    val similarityScore: Int,
    val documentId: Int,
)

/**
 * End-to-end query result including answer and persisted trace log location.
 *
 * @property answer Generated answer text.
 * @property runId Run identifier.
 * @property logs Absolute path to the written trace log file.
 */
data class QueryResult(
    val answer: String,
    val runId: String,
    val logs: String,
)

/**
 * Defines [ChatClient].
 */
interface ChatClient {
    /**
     * Generates a chat completion for the given system and user prompts.
     *
     * @param systemPrompt System prompt text.
     * @param userPrompt User prompt text.
     * @return Generated response text.
     */
    suspend fun complete(
        systemPrompt: String,
        userPrompt: String,
    ): String
}

/**
 * Implements [EchoChatClient].
 */
class EchoChatClient : ChatClient {
    /**
     * Returns a deterministic echo-style completion for local/demo usage.
     */
    override suspend fun complete(
        systemPrompt: String,
        userPrompt: String,
    ): String {
        val normalized = userPrompt.replace('\n', ' ')
        return "Based on the retrieved documents: ${normalized.take(220)}"
    }
}

/**
 * Defines [BaseRetriever].
 */
interface BaseRetriever {
    /**
     * Indexes the provided documents for retrieval.
     *
     * @param documents Document texts to index.
     */
    fun fit(documents: List<String>)

    /**
     * Retrieves up to [k] best-matching documents for the query.
     *
     * @param query User query text.
     * @param k Maximum number of results.
     */
    fun getTopK(
        query: String,
        k: Int = 3,
    ): List<Pair<Int, Int>>
}

/**
 * Implements [SimpleKeywordRetriever].
 */
class SimpleKeywordRetriever : BaseRetriever {
    private val documents = mutableListOf<String>()

    /**
     * Rebuilds the in-memory index from the provided documents.
     * @param documents Documents used to build the retriever index.
     */
    override fun fit(documents: List<String>) {
        this.documents.clear()
        this.documents.addAll(documents)
    }

    /**
     * Scores documents by keyword overlap and returns the top matches.
     */
    override fun getTopK(
        query: String,
        k: Int,
    ): List<Pair<Int, Int>> {
        val queryWords = query.lowercase().split(Regex("\\s+")).filter { it.isNotBlank() }
        return documents
            .mapIndexed { index, doc ->
                val documentWords = doc.lowercase().split(Regex("\\s+")).toSet()
                val matches = queryWords.count { word -> word in documentWords }
                index to matches
            }.sortedByDescending { (_, matches) -> matches }
            .take(k)
    }
}

/**
 * Implements [ExampleRag].
 *
 * @property chatClient Chat client used to generate responses.
 * @property retriever Retriever implementation.
 * @property systemPrompt System prompt template.
 * @property logDir Trace log directory.
 */
class ExampleRag(
    private val chatClient: ChatClient,
    private val retriever: BaseRetriever = SimpleKeywordRetriever(),
    private val systemPrompt: String =
        """
        Answer the following question based on the provided documents:
        Question: {query}
        Documents:
        {context}
        Answer:
        """.trimIndent(),
    private val logDir: String = "logs",
) {
    private val documents = mutableListOf<String>()
    private val traces = mutableListOf<TraceEvent>()
    private var isFitted = false

    init {
        File(logDir).mkdirs()
        traces +=
            TraceEvent(
                eventType = "init",
                component = "rag_system",
                data =
                    mapOf(
                        "retriever_type" to retriever::class.simpleName,
                        "system_prompt_length" to systemPrompt.length,
                        "logdir" to logDir,
                    ),
            )
    }

    /**
     * Appends new documents to the current corpus and re-fits the retriever.
     *
     * @param newDocuments Documents to append or replace.
     */
    fun addDocuments(newDocuments: List<String>) {
        traces +=
            TraceEvent(
                eventType = "document_operation",
                component = "rag_system",
                data =
                    mapOf(
                        "operation" to "add_documents",
                        "num_new_documents" to newDocuments.size,
                        "total_documents_before" to documents.size,
                        "document_lengths" to newDocuments.map { it.length },
                    ),
            )
        documents += newDocuments
        retriever.fit(documents)
        isFitted = true
    }

    /**
     * Replaces the current corpus with a new document set and re-fits the retriever.
     *
     * @param newDocuments Documents to set as the entire corpus.
     */
    fun setDocuments(newDocuments: List<String>) {
        documents.clear()
        documents += newDocuments
        retriever.fit(documents)
        isFitted = true
    }

    /**
     * Retrieves relevant documents for a query from the indexed corpus.
     *
     * @param query User query text.
     * @param topK Maximum documents to retrieve.
     */
    fun retrieveDocuments(
        query: String,
        topK: Int = 3,
    ): List<RetrievedDoc> {
        check(isFitted) { "No documents have been added. Call addDocuments() or setDocuments() first." }
        val topDocs = retriever.getTopK(query, topK)
        return topDocs
            .filter { (_, score) -> score > 0 }
            .map { (index, score) ->
                RetrievedDoc(
                    content = documents[index],
                    similarityScore = score,
                    documentId = index,
                )
            }
    }

    /**
     * Builds prompt context from retrieved documents and generates an answer.
     *
     * @param query User query text.
     * @param topK Maximum documents to retrieve.
     */
    suspend fun generateResponse(
        query: String,
        topK: Int = 3,
    ): String {
        val retrieved = retrieveDocuments(query, topK)
        if (retrieved.isEmpty()) {
            return "I couldn't find any relevant documents to answer your question."
        }

        val context =
            retrieved
                .mapIndexed { idx, doc -> "Document ${idx + 1}:\n${doc.content}" }
                .joinToString("\n\n")
        val prompt =
            systemPrompt
                .replace("{query}", query)
                .replace("{context}", context)

        return runCatching {
            chatClient.complete(systemPrompt = systemPrompt, userPrompt = prompt).trim()
        }.getOrElse { error ->
            "Error generating response: ${error.message}"
        }
    }

    /**
     * Runs an end-to-end RAG query and returns answer plus trace log path.
     *
     * @param question User question text.
     * @param topK Maximum documents to retrieve.
     * @param runId Run identifier.
     */
    suspend fun query(
        question: String,
        topK: Int = 3,
        runId: String? = null,
    ): QueryResult {
        val actualRunId = runId ?: generateRunId(question)
        traces.clear()
        traces +=
            TraceEvent(
                eventType = "query_start",
                component = "rag_system",
                data =
                    mapOf(
                        "run_id" to actualRunId,
                        "question" to question,
                        "question_length" to question.length,
                        "top_k" to topK,
                        "total_documents" to documents.size,
                    ),
            )

        return runCatching {
            val retrieved = retrieveDocuments(question, topK)
            val answer = generateResponse(question, topK)
            traces +=
                TraceEvent(
                    eventType = "query_complete",
                    component = "rag_system",
                    data =
                        mapOf(
                            "run_id" to actualRunId,
                            "success" to true,
                            "response_length" to answer.length,
                            "num_retrieved" to retrieved.size,
                        ),
                )
            val logs = exportTracesToLog(actualRunId, question, mapOf("answer" to answer, "run_id" to actualRunId))
            QueryResult(answer = answer, runId = actualRunId, logs = logs)
        }.getOrElse { error ->
            traces +=
                TraceEvent(
                    eventType = "error",
                    component = "rag_system",
                    data =
                        mapOf(
                            "run_id" to actualRunId,
                            "operation" to "query",
                            "error" to error.message,
                        ),
                )
            val logs = exportTracesToLog(actualRunId, question, null)
            QueryResult(
                answer = "Error processing query: ${error.message}",
                runId = actualRunId,
                logs = logs,
            )
        }
    }

    /**
     * Writes the current trace events to a JSONL log file for one run.
     *
     * @param runId Run identifier.
     * @param query User query text.
     * @param result Query result payload.
     */
    fun exportTracesToLog(
        runId: String,
        query: String?,
        result: Map<String, Any?>?,
    ): String {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)
        val filename = "rag_run_${runId}_${timestamp.replace(':', '-').replace('.', '-')}.jsonl"
        val filepath = File(logDir, filename)
        val logRows =
            listOf(
                mapOf(
                    "run_id" to runId,
                    "timestamp" to timestamp,
                    "query" to query,
                    "result" to result,
                    "num_documents" to documents.size,
                    "traces" to
                        traces.map { trace ->
                            mapOf(
                                "event_type" to trace.eventType,
                                "component" to trace.component,
                                "data" to trace.data,
                            )
                        },
                ),
            )
        filepath.parentFile.mkdirs()
        filepath.writeText(logRows.joinToString("\n") { row -> rowToJsonLine(row) } + "\n")
        return filepath.absolutePath
    }

    private fun generateRunId(question: String): String {
        val now = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val suffix = (question.hashCode().absoluteValue % 10_000).toString().padStart(4, '0')
        return "${now}_$suffix"
    }
}

/**
 * Creates a ready-to-use example RAG client with the default document corpus.
 *
 * @param chatClient Chat client implementation.
 * @param logDir Log output directory.
 */
fun defaultRagClient(
    chatClient: ChatClient,
    logDir: String = "logs",
): ExampleRag =
    ExampleRag(
        chatClient = chatClient,
        retriever = SimpleKeywordRetriever(),
        logDir = logDir,
    ).also { client ->
        client.addDocuments(DOCUMENTS)
    }
