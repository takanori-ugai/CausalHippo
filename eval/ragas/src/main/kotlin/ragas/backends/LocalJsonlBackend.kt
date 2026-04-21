package ragas.backends

import java.io.File
import java.nio.file.Files

/**
 * File-system backend that stores datasets and experiments as JSONL files.
 *
 * @property rootDir Root directory path.
 */
class LocalJsonlBackend(
    private val rootDir: String,
) : BaseBackend {
    private val validNamePattern = Regex("[A-Za-z0-9._-]+")

    private fun getDataDir(dataType: String): File = File(rootDir, dataType)

    private fun sanitizeName(name: String): String {
        require(name.matches(validNamePattern)) {
            "Invalid name '$name'. Only letters, digits, dot, underscore, and hyphen are allowed."
        }
        return name
    }

    private fun getFile(
        dataType: String,
        name: String,
    ): File = File(getDataDir(dataType), "${sanitizeName(name)}.jsonl")

    private fun load(
        dataType: String,
        name: String,
    ): List<Map<String, Any?>> {
        val file = getFile(dataType, name)
        if (!file.exists()) {
            throw java.io.FileNotFoundException("No ${dataType.dropLast(1)} named '$name' found at ${file.path}")
        }
        return file.useLines { lines ->
            lines
                .map { line -> line.trim() }
                .filter { line -> line.isNotEmpty() }
                .map { line -> jsonLineToRow(line) }
                .toList()
        }
    }

    private fun save(
        dataType: String,
        name: String,
        data: List<Map<String, Any?>>,
    ) {
        val file = getFile(dataType, name)
        file.parentFile.mkdirs()

        if (data.isEmpty()) {
            file.writeText("")
            return
        }

        Files.newBufferedWriter(file.toPath()).use { writer ->
            data.forEach { row ->
                writer.write(rowToJsonLine(row))
                writer.newLine()
            }
        }
    }

    private fun list(dataType: String): List<String> {
        val dir = getDataDir(dataType)
        if (!dir.exists()) {
            return emptyList()
        }

        return dir
            .listFiles { file -> file.isFile && file.extension == "jsonl" }
            ?.map { file -> file.nameWithoutExtension }
            ?.sorted()
            ?: emptyList()
    }

    /**
     * Loads a dataset by name from the backend.
     */
    override fun loadDataset(name: String): List<Map<String, Any?>> = load("datasets", name)

    /**
     * Loads an experiment by name from the backend.
     */
    override fun loadExperiment(name: String): List<Map<String, Any?>> = load("experiments", name)

    /**
     * Persists dataset rows under the provided dataset name.
     */
    override fun saveDataset(
        name: String,
        data: List<Map<String, Any?>>,
    ) {
        save("datasets", name, data)
    }

    /**
     * Persists experiment rows under the provided experiment name.
     */
    override fun saveExperiment(
        name: String,
        data: List<Map<String, Any?>>,
    ) {
        save("experiments", name, data)
    }

    /**
     * Lists available dataset names.
     */
    override fun listDatasets(): List<String> = list("datasets")

    /**
     * Lists available experiment names.
     */
    override fun listExperiments(): List<String> = list("experiments")

    /**
     * Returns a concise human-readable summary string.
     */
    override fun toString(): String = "LocalJsonlBackend(rootDir='$rootDir')"
}
