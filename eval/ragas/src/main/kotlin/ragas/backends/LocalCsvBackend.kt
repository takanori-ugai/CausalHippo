package ragas.backends

import java.io.File
import java.io.Reader

/**
 * File-system backend that stores datasets and experiments as CSV files.
 *
 * @property rootDir Root directory path.
 */
class LocalCsvBackend(
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
    ): File = File(getDataDir(dataType), "${sanitizeName(name)}.csv")

    private fun load(
        dataType: String,
        name: String,
    ): List<Map<String, Any?>> {
        val file = getFile(dataType, name)
        if (!file.exists()) {
            throw java.io.FileNotFoundException(
                "No ${dataType.dropLast(1)} named '$name' found at ${file.path}",
            )
        }
        return file.bufferedReader().use { reader ->
            val records = parseCsvRecords(reader)
            if (records.isEmpty()) {
                return@use emptyList()
            }

            val headers = records.first()
            records
                .drop(1)
                .filter { cells -> cells.any { cell -> cell.isNotBlank() } }
                .map { cells ->
                    headers
                        .mapIndexed { index, header ->
                            header to cells.getOrElse(index) { "" }
                        }.toMap()
                }
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

        val headers = data.first().keys.toList()
        val builder = StringBuilder()
        builder
            .append(headers.joinToString(",") { header -> csvEscape(header) })
            .append('\n')
        data.forEach { row ->
            val line =
                headers.joinToString(",") { key ->
                    csvEscape(row[key]?.toString().orEmpty())
                }
            builder.append(line).append('\n')
        }
        file.writeText(builder.toString())
    }

    private fun list(dataType: String): List<String> {
        val dir = getDataDir(dataType)
        if (!dir.exists()) {
            return emptyList()
        }

        return dir
            .listFiles { file -> file.isFile && file.extension == "csv" }
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
    override fun toString(): String = "LocalCsvBackend(rootDir='$rootDir')"

    private fun csvEscape(value: String): String {
        val needsQuotes = value.contains(',') || value.contains('"') || value.contains('\n')
        if (!needsQuotes) {
            return value
        }
        val escaped = value.replace("\"", "\"\"")
        return "\"$escaped\""
    }

    private fun parseCsvRecords(reader: Reader): List<List<String>> {
        val records = mutableListOf<List<String>>()
        var row = mutableListOf<String>()
        val current = StringBuilder()
        var inQuotes = false
        var chCode = reader.read()

        while (chCode != -1) {
            val ch = chCode.toChar()
            when {
                ch == '"' -> {
                    if (inQuotes) {
                        reader.mark(1)
                        val nextCode = reader.read()
                        if (nextCode == '"'.code) {
                            current.append('"')
                        } else {
                            inQuotes = false
                            if (nextCode != -1) {
                                reader.reset()
                            }
                        }
                    } else {
                        inQuotes = true
                    }
                }

                ch == ',' && !inQuotes -> {
                    row += current.toString()
                    current.clear()
                }

                ch == '\n' && !inQuotes -> {
                    row += current.toString()
                    records += row
                    row = mutableListOf()
                    current.clear()
                }

                ch == '\r' && !inQuotes -> {
                    reader.mark(1)
                    val nextCode = reader.read()
                    if (nextCode != '\n'.code && nextCode != -1) {
                        reader.reset()
                    }
                    row += current.toString()
                    records += row
                    row = mutableListOf()
                    current.clear()
                }

                else -> {
                    current.append(ch)
                }
            }
            chCode = reader.read()
        }

        require(!inQuotes) { "Malformed CSV: unterminated quoted field." }
        if (current.isNotEmpty() || row.isNotEmpty()) {
            row += current.toString()
            records += row
        }

        return records
    }
}
