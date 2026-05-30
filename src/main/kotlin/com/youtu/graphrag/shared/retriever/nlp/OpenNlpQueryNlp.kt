package com.youtu.graphrag.shared.retriever.nlp

import com.youtu.graphrag.shared.config.NlpConfig
import opennlp.tools.namefind.NameFinderME
import opennlp.tools.postag.POSModel
import opennlp.tools.postag.POSTaggerME
import opennlp.tools.tokenize.TokenizerME
import opennlp.tools.tokenize.TokenizerModel
import java.nio.file.Path
import java.util.Locale
import kotlin.io.path.exists
import kotlin.io.path.inputStream

class OpenNlpQueryNlp private constructor(
    private val tokenizer: TokenizerME,
    private val posTagger: POSTaggerME,
    private val entityFinders: List<NameFinderME>,
    stopwords: Collection<String>,
) : QueryNlp {
    private val stopwordSet =
        stopwords
            .asSequence()
            .map { word -> word.trim().lowercase(Locale.ROOT) }
            .filter { word -> word.isNotBlank() }
            .toSet()

    override fun analyze(question: String): QueryNlpAnalysis {
        val normalizedQuestion = question.trim()
        if (normalizedQuestion.isBlank()) {
            return QueryNlpAnalysis()
        }

        val tokens = tokenizer.tokenize(normalizedQuestion)
        if (tokens.isEmpty()) {
            return QueryNlpAnalysis()
        }

        val posTags = posTagger.tag(tokens)
        val entitySpans =
            entityFinders.flatMap { finder ->
                val spans = finder.find(tokens).toList()
                finder.clearAdaptiveData()
                spans
            }

        val entities =
            entitySpans
                .mapNotNull { span ->
                    val text = tokens.slice(span.start until span.end).joinToString(" ").trim()
                    text.takeIf { candidate -> candidate.length > 2 }
                }.distinctBy { entity -> entity.lowercase(Locale.ROOT) }

        val entityTokenIndexes =
            entitySpans
                .flatMap { span -> (span.start until span.end).toList() }
                .toSet()

        val keyTerms = mutableListOf<String>()
        val keywords = linkedSetOf<String>()

        tokens.forEachIndexed { index, token ->
            val normalized = token.lowercase(Locale.ROOT)
            if (normalized.length <= 2 || normalized in stopwordSet || normalized.all { it.isDigit() }) {
                return@forEachIndexed
            }

            val posTag = posTags.getOrElse(index) { "" }
            val eligibleByPos = posTag in KEYWORD_POS_TAGS
            val eligibleByEntity = index in entityTokenIndexes

            if (eligibleByPos || eligibleByEntity) {
                keywords.add(normalized)
                if (eligibleByPos && keyTerms.size < MAX_KEY_TERMS) {
                    keyTerms.add(token)
                }
            }
        }

        entities.forEach { entity ->
            val normalized = entity.lowercase(Locale.ROOT)
            if (normalized.length > 2) {
                keywords.add(normalized)
            }
        }

        return QueryNlpAnalysis(
            entities = entities,
            keyTerms = keyTerms,
            keywords = keywords.toList(),
        )
    }

    companion object {
        private const val MAX_KEY_TERMS = 5
        private val KEYWORD_POS_TAGS =
            setOf(
                "NN",
                "NNS",
                "NNP",
                "NNPS",
                "JJ",
                "JJR",
                "JJS",
                "VB",
                "VBD",
                "VBG",
                "VBN",
                "VBP",
                "VBZ",
            )

        fun fromConfig(
            config: NlpConfig,
            rootDir: Path,
        ): Result<OpenNlpQueryNlp> =
            runCatching {
                val tokenizerModelPath =
                    requireModelPath(
                        path = config.opennlp.tokenizerModelPath,
                        rootDir = rootDir,
                        label = "nlp.opennlp.tokenizer_model_path",
                    )
                val posModelPath =
                    requireModelPath(
                        path = config.opennlp.posModelPath,
                        rootDir = rootDir,
                        label = "nlp.opennlp.pos_model_path",
                    )
                val entityModelPaths =
                    listOfNotNull(
                        resolveOptionalModelPath(config.opennlp.personModelPath, rootDir),
                        resolveOptionalModelPath(config.opennlp.organizationModelPath, rootDir),
                        resolveOptionalModelPath(config.opennlp.locationModelPath, rootDir),
                    )

                val tokenizerModel = tokenizerModelPath.inputStream().use { stream -> TokenizerModel(stream) }
                val posModel = posModelPath.inputStream().use { stream -> POSModel(stream) }
                val entityFinders =
                    entityModelPaths.map { path ->
                        path.inputStream().use { stream ->
                            NameFinderME(opennlp.tools.namefind.TokenNameFinderModel(stream))
                        }
                    }

                OpenNlpQueryNlp(
                    tokenizer = TokenizerME(tokenizerModel),
                    posTagger = POSTaggerME(posModel),
                    entityFinders = entityFinders,
                    stopwords = config.stopwords,
                )
            }

        private fun requireModelPath(
            path: String,
            rootDir: Path,
            label: String,
        ): Path {
            val resolved = resolveOptionalModelPath(path, rootDir)
            requireNotNull(resolved) { "$label is required when nlp.provider=opennlp" }
            require(resolved.exists()) { "OpenNLP model file not found at $resolved" }
            return resolved
        }

        private fun resolveOptionalModelPath(
            path: String,
            rootDir: Path,
        ): Path? {
            val normalized = path.trim()
            if (normalized.isBlank()) {
                return null
            }
            val candidate = Path.of(normalized)
            return if (candidate.isAbsolute) candidate else rootDir.resolve(candidate).normalize()
        }
    }
}
