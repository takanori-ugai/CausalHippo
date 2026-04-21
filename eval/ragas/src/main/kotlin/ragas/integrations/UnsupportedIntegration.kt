package ragas.integrations

/**
 * Throws an error indicating that the integration is not supported yet.
 *
 * @param name Name or identifier.
 */
fun unsupportedIntegration(name: String): Nothing =
    throw UnsupportedOperationException(
        "Integration '$name' is not yet implemented in ragas-kotlin. Track parity in Plan.md.",
    )
