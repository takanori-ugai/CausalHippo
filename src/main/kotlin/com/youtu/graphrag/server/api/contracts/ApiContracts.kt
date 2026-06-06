package com.youtu.graphrag.server.api.contracts

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
data class FileUploadResponse(
    val success: Boolean,
    val message: String,
    @SerialName("dataset_name") val datasetName: String? = null,
    @SerialName("files_count") val filesCount: Int? = null,
)

@Serializable
data class GraphConstructionRequest(
    @SerialName("dataset_name") val datasetName: String,
)

@Serializable
data class GraphConstructionResponse(
    val success: Boolean,
    val message: String,
    @SerialName("graph_data") val graphData: JsonObject? = null,
)

@Serializable
data class QuestionRequest(
    val question: String,
    @SerialName("dataset_name") val datasetName: String,
)

@Serializable
data class ReasoningStep(
    val type: String,
    val question: String,
    val triples: List<String> = emptyList(),
    @SerialName("triples_count") val triplesCount: Int = 0,
    @SerialName("chunk_contents") val chunkContents: List<String> = emptyList(),
    @SerialName("chunks_count") val chunksCount: Int = 0,
    @SerialName("processing_time") val processingTime: Double = 0.0,
    val thought: String? = null,
)

@Serializable
data class QuestionResponse(
    val answer: String,
    @SerialName("sub_questions") val subQuestions: List<Map<String, String>> = emptyList(),
    @SerialName("retrieved_triples") val retrievedTriples: List<String> = emptyList(),
    @SerialName("retrieved_chunks") val retrievedChunks: List<String> = emptyList(),
    @SerialName("reasoning_steps") val reasoningSteps: List<ReasoningStep> = emptyList(),
    @SerialName("visualization_data") val visualizationData: JsonObject,
)

@Serializable
data class DatasetInfo(
    val name: String,
    val type: String,
    val status: String,
    @SerialName("has_custom_schema") val hasCustomSchema: Boolean,
)

@Serializable
data class DatasetsResponse(
    val datasets: List<DatasetInfo> = emptyList(),
)

@Serializable
data class BaseOperationResponse(
    val success: Boolean,
    val message: String,
    @SerialName("dataset_name") val datasetName: String? = null,
)

@Serializable
data class ProgressEvent(
    val type: String = "progress",
    val stage: String,
    val progress: Int,
    val message: String,
    val timestamp: String,
)

@Serializable
data class QaUpdateEvent(
    val type: String = "qa_update",
    val stage: String,
    val message: String? = null,
    val timestamp: String,
)

@Serializable
data class QaCompleteEvent(
    val type: String = "qa_complete",
    @SerialName("answer_preview") val answerPreview: String,
    val timestamp: String,
)
