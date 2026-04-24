package lightrag.examples

import kotlinx.coroutines.runBlocking
import lightrag.di.createLightRagRuntime

/**
 * The main function for the insert example.
 */
fun main() =
    runBlocking {
        val rag = createLightRagRuntime().rag

        val trackId = rag.insert("This is a test document content about Entity1 and Entity2.")
        println("Insert started with trackId: $trackId")

        // In a real app we would poll status, but here we just wait a bit or assume it's done if sync
        // The insert implementation calls pipelineProcessEnqueueDocuments() which runs in the same
        // coroutine scope in my implementation

        val status = rag.getProcessingStatus()
        println("Status: $status")
    }
