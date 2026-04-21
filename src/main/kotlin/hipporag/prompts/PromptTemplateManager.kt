@file:OptIn(ExperimentalSerializationApi::class)

package hipporag.prompts

import hipporag.utils.Message
import hipporag.utils.jsonWithDefaults
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.SerializationException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonClassDiscriminator
import java.io.File
import java.io.IOException
import java.net.JarURLConnection
import java.net.URL
import kotlin.OptIn

/**
 * Loads and renders prompt templates with simple variable substitution.
 */
class PromptTemplateManager(
    private val roleMapping: Map<String, String>,
) {
    private val logger = KotlinLogging.logger {}
    private val templates: Map<String, PromptTemplate> = PromptTemplates.templates

    /**
     * Renders a template named [name] with a single `prompt_user` variable.
     */
    fun render(
        name: String,
        promptUser: String,
    ): List<Message> = render(name, mapOf("prompt_user" to promptUser))

    /**
     * Renders a template named [name] with arbitrary [variables].
     */
    fun render(
        name: String,
        variables: Map<String, String>,
    ): List<Message> {
        val template = templates[name] ?: throw IllegalArgumentException("Template '$name' not found.")
        return when (template) {
            is PromptTemplate.Text -> {
                listOf(
                    Message(role = roleMapping["user"] ?: "user", content = substitute(template.template, variables)),
                )
            }

            is PromptTemplate.Chat -> {
                template.messages.map { msg ->
                    val role = roleMapping[msg.role] ?: msg.role
                    Message(role = role, content = substitute(msg.content, variables))
                }
            }
        }
    }

    /** Returns all available template names. */
    fun listTemplateNames(): List<String> = templates.keys.toList()

    /** Returns true if [name] exists in the template catalog. */
    fun isTemplateNameValid(name: String): Boolean = templates.containsKey(name)

    private fun substitute(
        template: String,
        variables: Map<String, String>,
    ): String {
        val pattern = Regex("""\$\{(\w+)\}""")
        return pattern.replace(template) { match ->
            val key = match.groupValues[1]
            variables[key] ?: run {
                logger.warn { "Missing prompt variable '$key'." }
                match.value
            }
        }
    }
}

/**
 * Sealed prompt template representation loaded from JSON resources.
 */
@Serializable
@JsonClassDiscriminator("type")
sealed class PromptTemplate {
    /**
     * Plain text template.
     *
     * @property template template string with placeholders.
     */
    @Serializable
    @SerialName("text")
    data class Text(
        val template: String,
    ) : PromptTemplate()

    /**
     * Multi-message chat template.
     *
     * @property messages ordered message templates.
     */
    @Serializable
    @SerialName("chat")
    data class Chat(
        val messages: List<PromptMessageTemplate>,
    ) : PromptTemplate()
}

/**
 * Template entry for an individual chat message.
 *
 * @property role message role (e.g., system/user/assistant).
 * @property content message content with variable placeholders.
 */
@Serializable
data class PromptMessageTemplate(
    val role: String,
    val content: String,
)

/**
 * Lazily-loaded prompt template catalog.
 */
object PromptTemplates {
    private val logger = KotlinLogging.logger {}

    /** Loaded templates keyed by name. */
    val templates: Map<String, PromptTemplate> by lazy {
        loadTemplates()
    }

    private val json =
        jsonWithDefaults {
            isLenient = true
            ignoreUnknownKeys = true
        }

    private fun loadTemplates(): Map<String, PromptTemplate> {
        val templatesPath = "prompts/templates"
        val classLoader = this::class.java.classLoader
        val templatesUrl = classLoader.getResource(templatesPath)
        if (templatesUrl == null) {
            logger.warn { "Templates directory '$templatesPath' not found in resources." }
            return emptyMap()
        }

        val templateFiles = listTemplateFiles(templatesUrl, templatesPath)
        if (templateFiles.isEmpty()) {
            logger.warn { "No template files found under '$templatesPath'." }
            return emptyMap()
        }

        return templateFiles
            .mapNotNull { fileName ->
                val resourcePath = "$templatesPath/$fileName"
                try {
                    val content =
                        classLoader
                            .getResourceAsStream(resourcePath)
                            ?.bufferedReader()
                            ?.use { it.readText() }
                            ?: run {
                                logger.warn { "Template resource '$resourcePath' not found." }
                                return@mapNotNull null
                            }
                    val templateName = fileName.substringBeforeLast(".json")
                    templateName to json.decodeFromString<PromptTemplate>(content)
                } catch (e: IOException) {
                    logger.error(e) { "Error loading template: $fileName" }
                    null
                } catch (e: SerializationException) {
                    logger.error(e) { "Error loading template: $fileName" }
                    null
                }
            }.toMap()
    }

    private fun listTemplateFiles(
        templatesUrl: URL,
        templatesPath: String,
    ): List<String> =
        when (templatesUrl.protocol) {
            "file" -> {
                val dir = File(templatesUrl.toURI())
                if (!dir.isDirectory) {
                    emptyList()
                } else {
                    dir
                        .listFiles { _, name -> name.endsWith(".json") }
                        ?.map { it.name }
                        ?: emptyList()
                }
            }

            "jar" -> {
                listTemplateFilesFromJar(templatesUrl, templatesPath)
            }

            else -> {
                logger.warn { "Unsupported templates URL protocol '${templatesUrl.protocol}'." }
                emptyList()
            }
        }

    private fun listTemplateFilesFromJar(
        templatesUrl: URL,
        templatesPath: String,
    ): List<String> {
        val connection = templatesUrl.openConnection() as? JarURLConnection
        val jarFile = connection?.jarFile ?: return emptyList()
        val entryPrefix =
            connection.entryName?.let { if (it.endsWith("/")) it else "$it/" }
                ?: "$templatesPath/"

        return jarFile
            .entries()
            .asSequence()
            .filter { !it.isDirectory && it.name.startsWith(entryPrefix) && it.name.endsWith(".json") }
            .map { it.name.removePrefix(entryPrefix) }
            .toList()
    }
}
