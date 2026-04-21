package data

fun main(args: Array<String>) {
    val defaults =
        OpenAlexToMusiqueCliArgs(
            inputPath = "data/openalex_eval_dataset.jsonl",
            outputPath = "data/openalex_eval_dataset_musique.jsonl",
        )
    val resolved = parseOpenAlexToMusiqueCliArgs(args, defaults)
    OpenAlexToMusiqueConverter().convert(
        inputPath = resolved.inputPath,
        outputPath = resolved.outputPath,
    )
}

private data class OpenAlexToMusiqueCliArgs(
    val inputPath: String,
    val outputPath: String,
)

private fun parseOpenAlexToMusiqueCliArgs(
    args: Array<String>,
    defaults: OpenAlexToMusiqueCliArgs,
): OpenAlexToMusiqueCliArgs {
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
                        throw IllegalArgumentException(
                            "Unknown argument: $arg. Supported options: --input, --output",
                        )
                    }

                    else -> {
                        positional += arg
                    }
                }
                i++
            }
        }
    }

    return OpenAlexToMusiqueCliArgs(
        inputPath = inputPath ?: positional.getOrNull(0) ?: defaults.inputPath,
        outputPath = outputPath ?: positional.getOrNull(1) ?: defaults.outputPath,
    )
}
