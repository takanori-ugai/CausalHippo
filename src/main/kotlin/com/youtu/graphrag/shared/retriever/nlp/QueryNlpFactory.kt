package com.youtu.graphrag.shared.retriever.nlp

import com.youtu.graphrag.shared.config.ConfigManager
import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Path
import java.util.Locale

object QueryNlpFactory {
    private val logger = KotlinLogging.logger {}

    fun create(
        config: ConfigManager,
        rootDir: Path = Path.of("."),
    ): QueryNlp {
        val provider =
            config.nlp.provider
                .trim()
                .lowercase(Locale.ROOT)
        val fallback = RegexQueryNlp(config.nlp.stopwords)

        return when (provider) {
            "opennlp",
            "open_nlp",
            -> {
                OpenNlpQueryNlp
                    .fromConfig(config.nlp, rootDir)
                    .getOrElse { error ->
                        logger.warn(error) {
                            "OpenNLP provider could not be initialized; falling back to regex query NLP."
                        }
                        fallback
                    }
            }

            "regex",
            "",
            -> {
                fallback
            }

            else -> {
                logger.warn { "Unknown nlp.provider '$provider'; falling back to regex query NLP." }
                fallback
            }
        }
    }
}
