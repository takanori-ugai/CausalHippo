package ragas.prompt

import java.io.File

/**
 * Mutable prompt collection contract with persistence helpers.
 */
interface PromptCollection {
    /**
     * Returns the current map of prompt definitions.
     */
    fun getPrompts(): Map<String, SimplePrompt>

    /**
     * Replaces prompts using the provided map of definitions.
     *
     * @param prompts Prompt map to set on the collection.
     */
    fun setPrompts(prompts: Map<String, SimplePrompt>)

    /**
     * Saves prompt definitions to the provided path.
     *
     * @param path Filesystem path.
     */
    fun savePrompts(path: String) {
        val dir = File(path)
        require(dir.mkdirs() || dir.isDirectory) { "Failed to create directory: $path" }
        getPrompts().forEach { (name, prompt) ->
            prompt.save(File(dir, "${name}_${prompt.language}.json").path)
        }
    }

    /**
     * Loads prompt definitions from the provided path.
     *
     * @param path Filesystem path.
     * @param language Target language for prompt loading/adaptation.
     */
    fun loadPrompts(
        path: String,
        language: String = "english",
    ): Map<String, SimplePrompt> {
        val dir = File(path)
        require(dir.isDirectory) { "Path is not a directory: $path" }

        return getPrompts().keys.associateWith { name ->
            val languagePath = File(dir, "${name}_$language.json")
            val legacyPath = File(dir, "$name.json")
            when {
                languagePath.exists() -> SimplePrompt.load(languagePath.path)
                legacyPath.exists() -> SimplePrompt.load(legacyPath.path)
                else -> SimplePrompt.load(languagePath.path)
            }
        }
    }
}
