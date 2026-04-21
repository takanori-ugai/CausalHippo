package data

import causalrag.generator.llm.LLMInterface
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.apache.pdfbox.Loader
import org.apache.pdfbox.text.PDFTextStripper
import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.time.Duration
import java.util.Locale

fun main(args: Array<String>) {
    val cli = parseCliArgs(args)
    val builder = OpenAlexDatasetBuilder(cli)
    builder.run()
}

private data class OpenAlexCliArgs(
    val outputPath: String,
    val maxDocs: Int,
    val questionsPerDoc: Int,
    val modelName: String,
    val provider: String,
    val apiKey: String?,
    val openAlexApiKey: String?,
    val openAlexEmail: String?,
    val perPage: Int,
    val maxPages: Int,
)

private data class OpenAlexWork(
    val id: String,
    val title: String,
    val abstract: String,
    val introduction: String,
    val fullText: String,
    val textForQuestions: String,
    val textSource: String,
    val sharedEvidenceUnits: List<String>,
    val field: String,
    val publicationYear: Int?,
    val workType: String?,
    val doi: String?,
    val citedByCount: Int?,
)

@Serializable
private data class DatasetRecord(
    val id: String,
    val title: String,
    val field: String,
    val publicationYear: Int?,
    val workType: String?,
    val doi: String?,
    val citedByCount: Int?,
    val abstract: String,
    val introduction: String,
    val fullText: String,
    val textSource: String,
    val text: String,
    val questions: List<SectionQuestion>,
)

@Serializable
private data class LlmQuestionResponse(
    val questions: List<SectionQuestion> = emptyList(),
)

@Serializable
private data class SectionQuestion(
    val question: String,
    val source: String,
)

private class OpenAlexDatasetBuilder(
    private val cli: OpenAlexCliArgs,
) {
    private val httpClient: HttpClient = HttpClient.newHttpClient()
    private val json =
        Json {
            ignoreUnknownKeys = true
            prettyPrint = false
        }
    private val llm: LLMInterface by lazy {
        LLMInterface(
            modelName = cli.modelName,
            apiKey = cli.apiKey,
            provider = cli.provider,
            systemMessage =
                "You generate factual evaluation questions from scientific text. " +
                    "Follow section constraints exactly and output strict JSON.",
        )
    }

    fun run() {
        require(cli.maxDocs > 0) { "--max-docs must be > 0" }
        require(cli.questionsPerDoc > 0) { "--questions-per-doc must be > 0" }
        require(cli.perPage in 1..200) { "--per-page must be between 1 and 200" }
        require(cli.maxPages > 0) { "--max-pages must be > 0" }

        println("Fetching OpenAlex works (target=${cli.maxDocs}, perPage=${cli.perPage}, maxPages=${cli.maxPages})...")
        val works = fetchWorks()
        if (works.isEmpty()) {
            error("No valid works found from OpenAlex with current filters.")
        }

        val stratified = stratifiedRoundRobinSample(works, cli.maxDocs)
        println("Selected ${stratified.size} documents after field-balanced sampling.")

        val outPath = Path.of(cli.outputPath)
        Files.createDirectories(outPath.parent ?: Path.of("."))

        outPath.toFile().bufferedWriter().use { writer ->
            stratified.forEachIndexed { index, work ->
                println("[${index + 1}/${stratified.size}] Generating questions for ${work.id}")
                val questions = generateQuestions(work)
                val record =
                    DatasetRecord(
                        id = work.id,
                        title = work.title,
                        field = work.field,
                        publicationYear = work.publicationYear,
                        workType = work.workType,
                        doi = work.doi,
                        citedByCount = work.citedByCount,
                        abstract = work.abstract,
                        introduction = work.introduction,
                        fullText = work.fullText,
                        textSource = work.textSource,
                        text = work.textForQuestions,
                        questions = questions,
                    )
                writer.append(json.encodeToString(record))
                writer.newLine()
            }
        }

        println("Done. Wrote ${stratified.size} records to ${outPath.toAbsolutePath()}")
    }

    private fun fetchWorks(): List<OpenAlexWork> {
        val works = mutableListOf<OpenAlexWork>()
        var cursor = "*"
        var page = 0

        while (page < cli.maxPages) {
            page += 1
            val url = buildOpenAlexUrl(cursor)
            val body = httpGet(url)
            val root = json.parseToJsonElement(body).jsonObject
            val results = root["results"]?.jsonArray ?: JsonArray(emptyList())
            println("Fetched page $page with ${results.size} results")

            for (result in results) {
                parseWork(result.jsonObject)?.let { works += it }
            }

            val nextCursor = root["meta"]?.jsonObject?.get("next_cursor")?.asStringOrNull()
            if (nextCursor.isNullOrBlank()) {
                break
            }

            cursor = nextCursor

            val enoughForBalancing = works.size >= cli.maxDocs * 5
            if (enoughForBalancing) {
                break
            }
        }

        return works
    }

    private fun buildOpenAlexUrl(cursor: String): String {
        val filter = "has_abstract:true,type:article|proceedings-article,language:en"
        val selectFields =
            listOf(
                "id",
                "title",
                "abstract_inverted_index",
                "publication_year",
                "type",
                "doi",
                "cited_by_count",
                "concepts",
                "primary_topic",
                "best_oa_location",
                "primary_location",
                "locations",
            ).joinToString(",")
        val params =
            linkedMapOf(
                "filter" to filter,
                "per-page" to cli.perPage.toString(),
                "cursor" to cursor,
                "select" to selectFields,
            )

        if (!cli.openAlexEmail.isNullOrBlank()) {
            params["mailto"] = cli.openAlexEmail
        }
        if (!cli.openAlexApiKey.isNullOrBlank()) {
            params["api_key"] = cli.openAlexApiKey
        }

        val query =
            params.entries.joinToString("&") { (k, v) ->
                "${urlEncode(k)}=${urlEncode(v)}"
            }

        return "https://api.openalex.org/works?$query"
    }

    private fun parseWork(obj: JsonObject): OpenAlexWork? {
        val id = obj["id"]?.asStringOrNull() ?: return null
        val title = obj["title"]?.asStringOrNull()?.trim().orEmpty()
        val abstractObj = obj["abstract_inverted_index"]?.jsonObject ?: return null
        val abstract = reconstructAbstract(abstractObj)
        if (abstract.isBlank()) return null
        if (!hasAvailablePdfUrl(obj)) return null
        val resolvedFullText = resolveFullText(obj, abstract)
        val fullText = resolvedFullText ?: abstract
        if (countWords(fullText) < 1000) return null
        val introduction = extractIntroduction(fullText, abstract)
        if (abstract.length > introduction.length) return null
        val sharedEvidenceUnits = extractSharedEvidenceUnitsWithFallback(abstract, introduction)
        if (sharedEvidenceUnits.isEmpty()) return null
        val textForQuestions = sharedEvidenceUnits.joinToString("\n")
        val textSource =
            if (resolvedFullText != null) {
                "abstract_introduction_overlap"
            } else {
                "abstract_fallback"
            }

        return OpenAlexWork(
            id = id,
            title = title.ifBlank { "Untitled" },
            abstract = abstract,
            introduction = introduction,
            fullText = fullText,
            textForQuestions = textForQuestions,
            textSource = textSource,
            sharedEvidenceUnits = sharedEvidenceUnits,
            field = detectField(obj),
            publicationYear = obj["publication_year"]?.asIntOrNull(),
            workType = obj["type"]?.asStringOrNull(),
            doi = obj["doi"]?.asStringOrNull(),
            citedByCount = obj["cited_by_count"]?.asIntOrNull(),
        )
    }

    private fun resolveFullText(
        obj: JsonObject,
        abstract: String,
    ): String? {
        val candidates = candidateOpenAccessUrls(obj)
        for (url in candidates) {
            val fetched = fetchReadableText(url) ?: continue
            if (!looksLikeSubstantialPaperText(fetched, abstract)) continue
            return fetched
        }
        return null
    }

    private fun candidateOpenAccessUrls(obj: JsonObject): List<String> {
        val urls = linkedSetOf<String>()
        val locationObjects = mutableListOf<JsonObject>()

        obj["best_oa_location"]?.asJsonObjectOrNull()?.let { locationObjects += it }
        obj["primary_location"]?.asJsonObjectOrNull()?.let { locationObjects += it }
        val locations = obj["locations"]?.jsonArray ?: JsonArray(emptyList())
        locations.forEach { element ->
            element.asJsonObjectOrNull()?.let { locationObjects += it }
        }

        for (location in locationObjects) {
            location["landing_page_url"]
                ?.asStringOrNull()
                ?.trim()
                ?.takeIf { it.isNotBlank() }
                ?.let { urls += it }
            location["pdf_url"]
                ?.asStringOrNull()
                ?.trim()
                ?.takeIf { it.isNotBlank() }
                ?.let { urls += it }
        }

        return urls.toList()
    }

    private fun hasAvailablePdfUrl(obj: JsonObject): Boolean {
        val locationObjects = mutableListOf<JsonObject>()
        obj["best_oa_location"]?.asJsonObjectOrNull()?.let { locationObjects += it }
        obj["primary_location"]?.asJsonObjectOrNull()?.let { locationObjects += it }
        val locations = obj["locations"]?.jsonArray ?: JsonArray(emptyList())
        locations.forEach { element ->
            element.asJsonObjectOrNull()?.let { locationObjects += it }
        }

        return locationObjects.any { location ->
            location["pdf_url"]
                ?.asStringOrNull()
                ?.trim()
                ?.isNotBlank() == true
        }
    }

    private fun fetchReadableText(url: String): String? {
        if (url.isBlank()) return null

        return try {
            val request =
                HttpRequest
                    .newBuilder()
                    .uri(URI(url))
                    .header("Accept", "application/pdf,text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8")
                    .header("User-Agent", buildUserAgent())
                    .timeout(Duration.ofSeconds(15))
                    .GET()
                    .build()

            val response = httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray())
            if (response.statusCode() !in 200..299) return null

            val bodyBytes = response.body()
            if (bodyBytes.isEmpty()) return null
            val contentType =
                response
                    .headers()
                    .firstValue("content-type")
                    .orElse("")
                    .lowercase()

            val text =
                if (isLikelyPdfUrl(url) || contentType.contains("application/pdf")) {
                    extractPdfText(bodyBytes) ?: return null
                } else {
                    val bodyText = String(bodyBytes, StandardCharsets.UTF_8)
                    if (contentType.contains("text/plain")) bodyText else htmlToText(bodyText)
                }

            normalizeText(text)
        } catch (_: Exception) {
            null
        }
    }

    private fun isLikelyPdfUrl(url: String): Boolean {
        val normalizedUrl = url.lowercase(Locale.US)
        return normalizedUrl.endsWith(".pdf") || normalizedUrl.contains("/pdf") || normalizedUrl.contains("format=pdf")
    }

    private fun extractPdfText(bytes: ByteArray): String? {
        return try {
            Loader.loadPDF(bytes).use { document ->
                if (document.numberOfPages <= 0) return null
                val stripper = PDFTextStripper()
                val raw = stripper.getText(document)
                normalizeText(raw)
            }
        } catch (_: Exception) {
            null
        }
    }

    private fun htmlToText(html: String): String {
        if (html.isBlank()) return ""
        return html
            .replace("(?is)<script\\b.*?</script>".toRegex(), " ")
            .replace("(?is)<style\\b.*?</style>".toRegex(), " ")
            .replace("(?is)<noscript\\b.*?</noscript>".toRegex(), " ")
            .replace("(?is)<[^>]+>".toRegex(), " ")
            .replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
    }

    private fun normalizeText(text: String): String =
        text
            .replace("\\s+".toRegex(), " ")
            .trim()

    private fun countWords(text: String): Int = "[A-Za-z0-9]+".toRegex().findAll(text).count()

    private fun looksLikeSubstantialPaperText(
        text: String,
        abstract: String,
    ): Boolean {
        if (text.length < 1800) return false
        if (text.length < abstract.length * 2) return false
        return true
    }

    private fun extractIntroduction(
        fullText: String,
        abstract: String,
    ): String {
        val source = if (fullText.isNotBlank()) fullText else abstract
        if (source.isBlank()) return ""

        val introHeader = "(?i)\\b(1\\.?\\s*)?introduction\\b"
        val introMatch = introHeader.toRegex().find(source)
        if (introMatch != null) {
            val start = introMatch.range.first
            val tail = source.substring(start).trim()
            val sectionEnd =
                "(?i)\\b(2\\.?\\s*)?(materials and methods|methods|methodology|background|results|related work|conclusion)\\b"
                    .toRegex()
                    .find(
                        tail,
                        startIndex = 20,
                    )?.range
                    ?.first
            val clipped = if (sectionEnd != null) tail.substring(0, sectionEnd) else tail.take(2200)
            return normalizeText(clipped)
        }

        val sentences =
            source
                .split(Regex("(?<=[.!?])\\s+"))
                .map { it.trim() }
                .filter { it.isNotBlank() }
                .take(4)
                .joinToString(" ")
        return normalizeText(sentences).take(2200)
    }

    private fun detectField(obj: JsonObject): String {
        val primaryTopicField =
            obj["primary_topic"]
                ?.jsonObject
                ?.get("field")
                ?.jsonObject
                ?.get("display_name")
                ?.asStringOrNull()
        if (!primaryTopicField.isNullOrBlank()) {
            return primaryTopicField
        }

        val concepts = obj["concepts"]?.jsonArray ?: return "Other"
        val topLevel =
            concepts
                .mapNotNull { element ->
                    val concept = element.jsonObject
                    val level = concept["level"]?.asIntOrNull()
                    if (level != 0) return@mapNotNull null
                    val score = concept["score"]?.asDoubleOrNull() ?: 0.0
                    val name = concept["display_name"]?.asStringOrNull() ?: return@mapNotNull null
                    score to name
                }.maxByOrNull { it.first }
                ?.second

        return topLevel ?: "Other"
    }

    private fun reconstructAbstract(invertedIndex: JsonObject): String {
        val positionedTokens = mutableListOf<Pair<Int, String>>()
        for ((token, positionsElement) in invertedIndex) {
            val positions = positionsElement as? JsonArray ?: continue
            for (pos in positions) {
                val index = pos.asIntOrNull() ?: continue
                positionedTokens += index to token
            }
        }

        if (positionedTokens.isEmpty()) return ""

        positionedTokens.sortBy { it.first }
        return positionedTokens.joinToString(" ") { it.second }.replace("\\s+".toRegex(), " ").trim()
    }

    private fun generateQuestions(work: OpenAlexWork): List<SectionQuestion> {
        val abstractTarget = (cli.questionsPerDoc + 1) / 2
        val introTarget = cli.questionsPerDoc / 2
        val llmQuestions =
            runCatching {
                generateQuestionsWithLlm(
                    work = work,
                    abstractTarget = abstractTarget,
                    introTarget = introTarget,
                )
            }.getOrElse { ex ->
                val detail = ex.message ?: ex::class.simpleName ?: "unknown error"
                println("LLM question generation failed for ${work.id}: $detail")
                emptyList()
            }
        return llmQuestions
    }

    private fun generateQuestionsWithLlm(
        work: OpenAlexWork,
        abstractTarget: Int,
        introTarget: Int,
    ): List<SectionQuestion> {
        val prompt =
            """
            Create evaluation questions for the paper below.
            Return strict JSON with this exact schema:
            {
              "questions": [
                {"question": "question 1", "source": "abstract"},
                {"question": "question 2", "source": "introduction"}
              ]
            }
            
            Requirements:
            - Each question must be answerable when only one section is provided.
            - Every item must have "source" set to either "abstract" or "introduction".
            - If "source" is "abstract", that question must be answerable from only Abstract.
            - If "source" is "introduction", that question must be answerable from only Introduction.
            - Do not require combining evidence from both sections.
            - Do not use yes/no questions.
            - Keep questions short, factual, and specific.
            - Do not mention "abstract" or "introduction" in the question text.
            - Output JSON only, no markdown.
            - Generate up to $abstractTarget questions with source "abstract".
            - Generate up to $introTarget questions with source "introduction".
            
            Title:
            ${work.title}
            
            Abstract:
            ${work.abstract}
            
            Introduction:
            ${work.introduction}
            """.trimIndent()

        val raw = llm.generate(prompt = prompt, temperature = 0.2, maxTokens = 1_200, jsonMode = true)
        val parsed = json.decodeFromString(LlmQuestionResponse.serializer(), extractFirstJsonObject(raw))

        val normalizedQuestions =
            parsed.questions
                .mapNotNull { item ->
                    val clean = cleanQuestion(item.question) ?: return@mapNotNull null
                    val source = item.source.lowercase(Locale.US).trim()
                    if (source != "abstract" && source != "introduction") return@mapNotNull null
                    SectionQuestion(question = clean, source = source)
                }.distinctBy { "${it.source}::${it.question}" }

        val abstractQuestions = normalizedQuestions.filter { it.source == "abstract" }
        val introQuestions = normalizedQuestions.filter { it.source == "introduction" }

        if (abstractQuestions.isEmpty() && introQuestions.isEmpty()) {
            error("LLM returned no usable questions. Raw response prefix: ${raw.take(400)}")
        }

        return interleaveQuestions(
            abstractQuestions = abstractQuestions.take(abstractTarget),
            introQuestions = introQuestions.take(introTarget),
            target = cli.questionsPerDoc,
        )
    }

    private fun interleaveQuestions(
        abstractQuestions: List<SectionQuestion>,
        introQuestions: List<SectionQuestion>,
        target: Int,
    ): List<SectionQuestion> {
        val out = mutableListOf<SectionQuestion>()
        val a = abstractQuestions.iterator()
        val i = introQuestions.iterator()
        while (out.size < target && (a.hasNext() || i.hasNext())) {
            if (a.hasNext()) out += a.next()
            if (out.size >= target) break
            if (i.hasNext()) out += i.next()
        }
        return out.take(target)
    }

    private fun extractSharedEvidenceUnits(
        abstract: String,
        introduction: String,
    ): List<String> {
        val normalizedIntro = normalizeForMatching(introduction)
        val sentenceMatches = mutableListOf<String>()
        splitSentences(abstract).forEach { sentence ->
            val normalizedSentence = normalizeForMatching(sentence)
            if (normalizedSentence.length < 50) return@forEach
            if (normalizedIntro.contains(normalizedSentence)) {
                sentenceMatches += normalizeText(sentence)
            }
        }
        if (sentenceMatches.isNotEmpty()) {
            return sentenceMatches.distinct().take(cli.questionsPerDoc * 3)
        }

        val abstractTokens = tokenizeForOverlap(abstract)
        val introTokens = tokenizeForOverlap(introduction)
        if (abstractTokens.size < OVERLAP_NGRAM_SIZE || introTokens.size < OVERLAP_NGRAM_SIZE) {
            return emptyList()
        }

        val abstractNgrams =
            abstractTokens
                .windowed(size = OVERLAP_NGRAM_SIZE, step = 1, partialWindows = false)
                .map { it.joinToString(" ") }
                .toSet()
        val overlap = mutableListOf<String>()
        introTokens
            .windowed(size = OVERLAP_NGRAM_SIZE, step = 1, partialWindows = false)
            .forEach { ngramTokens ->
                val ngram = ngramTokens.joinToString(" ")
                if (ngram in abstractNgrams) {
                    overlap += ngram
                }
            }
        return overlap.distinct().take(cli.questionsPerDoc * 5)
    }

    private fun extractSharedEvidenceUnitsWithFallback(
        abstract: String,
        introduction: String,
    ): List<String> {
        val overlapUnits = extractSharedEvidenceUnits(abstract, introduction)
        if (overlapUnits.isNotEmpty()) {
            return overlapUnits
        }

        // Fallback when full-text retrieval fails or introduction extraction is weak.
        return splitSentences(abstract)
            .map { normalizeText(it) }
            .filter { it.length >= 50 }
            .distinct()
            .take(cli.questionsPerDoc * 3)
    }

    private fun splitSentences(text: String): List<String> =
        text
            .split(Regex("(?<=[.!?])\\s+"))
            .map { it.trim() }
            .filter { it.isNotBlank() }

    private fun normalizeForMatching(text: String): String =
        text
            .lowercase(Locale.US)
            .replace("[^a-z0-9\\s]".toRegex(), " ")
            .replace("\\s+".toRegex(), " ")
            .trim()

    private fun tokenizeForOverlap(text: String): List<String> =
        normalizeForMatching(text)
            .split(" ")
            .filter { token -> token.length >= 4 && token !in OVERLAP_STOP_WORDS }

    private fun stratifiedRoundRobinSample(
        works: List<OpenAlexWork>,
        targetCount: Int,
    ): List<OpenAlexWork> {
        val byField = works.groupBy { it.field }.mapValues { it.value.toMutableList() }.toMutableMap()
        val fields = byField.keys.sortedByDescending { byField[it]?.size ?: 0 }
        val selected = mutableListOf<OpenAlexWork>()

        while (selected.size < targetCount && byField.isNotEmpty()) {
            var addedInRound = 0
            for (field in fields) {
                val bucket = byField[field] ?: continue
                if (bucket.isEmpty()) {
                    byField.remove(field)
                    continue
                }
                val next = bucket.removeFirst()
                selected += next
                addedInRound += 1
                if (selected.size >= targetCount) {
                    break
                }
            }
            if (addedInRound == 0) {
                break
            }
        }

        val distribution =
            selected
                .groupingBy { it.field }
                .eachCount()
                .entries
                .sortedByDescending { it.value }
        println("Selected field distribution:")
        distribution.forEach { (field, count) ->
            val pct = if (selected.isEmpty()) 0.0 else (100.0 * count / selected.size)
            println("- $field: $count (${String.format(Locale.US, "%.1f", pct)}%)")
        }

        return selected
    }

    private fun httpGet(url: String): String {
        val requestBuilder =
            HttpRequest
                .newBuilder()
                .uri(URI(url))
                .header("Accept", "application/json")
                .header("User-Agent", buildUserAgent())
                .timeout(Duration.ofSeconds(20))
                .GET()

        val request = requestBuilder.build()
        val response = httpClient.send(request, HttpResponse.BodyHandlers.ofString())
        if (response.statusCode() !in 200..299) {
            error("OpenAlex request failed (${response.statusCode()}): ${response.body().take(500)}")
        }
        return response.body()
    }

    private fun buildUserAgent(): String {
        val base = "Hybrid2-OpenAlexDatasetBuilder/1.0"
        val email = cli.openAlexEmail
        return if (email.isNullOrBlank()) base else "$base (mailto:$email)"
    }

    private fun cleanQuestion(raw: String): String? {
        val normalized = normalizeText(raw)
        if (normalized.isBlank()) return null
        val withoutListPrefix = normalized.replace("^[0-9]+[.)]\\s*".toRegex(), "")
        val q = if (withoutListPrefix.endsWith("?")) withoutListPrefix else "$withoutListPrefix?"
        return q.takeIf { it.length >= 12 }
    }

    private fun extractFirstJsonObject(raw: String): String {
        val trimmed = raw.trim()
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            return trimmed
        }
        val start = trimmed.indexOf('{')
        val end = trimmed.lastIndexOf('}')
        require(start >= 0 && end > start) { "LLM response did not contain a JSON object." }
        return trimmed.substring(start, end + 1)
    }

    private fun urlEncode(value: String): String = URLEncoder.encode(value, StandardCharsets.UTF_8)
}

private const val OVERLAP_NGRAM_SIZE = 10

private val OVERLAP_STOP_WORDS =
    setOf(
        "about",
        "after",
        "also",
        "among",
        "between",
        "from",
        "have",
        "into",
        "more",
        "most",
        "other",
        "over",
        "such",
        "than",
        "that",
        "their",
        "there",
        "these",
        "this",
        "those",
        "through",
        "under",
        "using",
        "with",
        "within",
    )

private fun parseCliArgs(args: Array<String>): OpenAlexCliArgs {
    val defaults =
        OpenAlexCliArgs(
            outputPath = "data/openalex_eval_dataset.jsonl",
            maxDocs = 20,
            questionsPerDoc = 5,
            modelName = "gpt-5.4-mini",
            provider = "openai",
            apiKey = null,
            openAlexApiKey = System.getenv("OPENALEX_API_KEY"),
            openAlexEmail = null,
            perPage = 100,
            maxPages = 10,
        )

    val options = mutableMapOf<String, String>()
    var i = 0
    while (i < args.size) {
        val arg = args[i]
        require(arg.startsWith("--")) { "Unknown positional argument: $arg" }
        val key = arg.removePrefix("--")
        val value = args.getOrNull(i + 1)
        require(value != null && !value.startsWith("--")) { "Missing value for --$key" }
        options[key] = value
        i += 2
    }

    return OpenAlexCliArgs(
        outputPath = options["output"] ?: defaults.outputPath,
        maxDocs = (options["max-docs"] ?: defaults.maxDocs.toString()).toInt(),
        questionsPerDoc = (options["questions-per-doc"] ?: defaults.questionsPerDoc.toString()).toInt(),
        modelName = options["model"] ?: defaults.modelName,
        provider = options["provider"] ?: defaults.provider,
        apiKey = options["api-key"],
        openAlexApiKey = options["openalex-api-key"] ?: defaults.openAlexApiKey,
        openAlexEmail = options["openalex-email"],
        perPage = (options["per-page"] ?: defaults.perPage.toString()).toInt(),
        maxPages = (options["max-pages"] ?: defaults.maxPages.toString()).toInt(),
    )
}

private fun JsonElement.asStringOrNull(): String? = (this as? JsonPrimitive)?.contentOrNull

private fun JsonElement.asIntOrNull(): Int? = (this as? JsonPrimitive)?.contentOrNull?.toIntOrNull()

private fun JsonElement.asDoubleOrNull(): Double? = (this as? JsonPrimitive)?.contentOrNull?.toDoubleOrNull()

private fun JsonElement.asJsonObjectOrNull(): JsonObject? = this as? JsonObject
