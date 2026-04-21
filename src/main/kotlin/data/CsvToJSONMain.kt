package data

fun main(args: Array<String>) {
    val defaults =
        CliArgs(
            csvPath = "data/Data-Common_Sense_Causation.csv",
            outputPath = "data/Data-Common_Sense_Causation.jsonl",
        )
    val (csvPath, outputPath) = parseCliArgs(args, defaults)

    println("Converting CSV to JSONL: $csvPath -> $outputPath")
    CsvToJsonConverter().convert(csvPath, outputPath)
}

private data class CliArgs(
    val csvPath: String,
    val outputPath: String,
)

private fun parseCliArgs(
    args: Array<String>,
    defaults: CliArgs,
): CliArgs {
    var inputPath: String? = null
    var outputPath: String? = null
    val positional = mutableListOf<String>()
    var i = 0

    while (i < args.size) {
        when (val arg = args[i]) {
            "--input" -> {
                require(i + 1 < args.size) { "Missing value for --input" }
                inputPath = args[i + 1]
                i += 2
            }

            "--output" -> {
                require(i + 1 < args.size) { "Missing value for --output" }
                outputPath = args[i + 1]
                i += 2
            }

            else -> {
                when {
                    arg.startsWith("--input=") -> {
                        inputPath = arg.substringAfter("=")
                    }

                    arg.startsWith("--output=") -> {
                        outputPath = arg.substringAfter("=")
                    }

                    arg.startsWith("--") -> {
                        throw IllegalArgumentException("Unknown argument: $arg. Supported options: --input, --output")
                    }

                    else -> {
                        positional += arg
                    }
                }
                i++
            }
        }
    }

    val resolvedInput = inputPath ?: positional.getOrNull(0) ?: defaults.csvPath
    val resolvedOutput = outputPath ?: positional.getOrNull(1) ?: defaults.outputPath
    return CliArgs(csvPath = resolvedInput, outputPath = resolvedOutput)
}
