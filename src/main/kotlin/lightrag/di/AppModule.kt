package lightrag.di

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.Encoding
import com.knuddels.jtokkit.api.EncodingRegistry
import com.knuddels.jtokkit.api.EncodingType
import com.typesafe.config.Config
import com.typesafe.config.ConfigFactory
import dev.langchain4j.model.chat.ChatModel
import dev.langchain4j.model.chat.StreamingChatModel
import dev.langchain4j.model.embedding.EmbeddingModel
import lightrag.core.AddonConfig
import lightrag.core.LightRAG
import lightrag.core.LightRagOverrides
import lightrag.core.Neo4jConfig
import lightrag.llm.DualChatModel
import lightrag.llm.LLMFactory
import lightrag.services.IngestionService
import lightrag.services.QueryService
import lightrag.services.StorageManager
import org.koin.core.qualifier.named
import org.koin.dsl.module

val appModule =
    module {
        single {
            val rawConfig: Config = ConfigFactory.load()
            val lightragConfig = rawConfig.getConfig("lightrag")
            val openaiConfig = lightragConfig.getConfig("openai")
            val ollamaConfig = lightragConfig.getConfig("ollama")
            val neo4jConfig = lightragConfig.getConfig("neo4j")
            val mongodbConfig = lightragConfig.getConfig("mongodb")
            val storageConfig = lightragConfig.getConfig("storage")
            val addonConfigValues = lightragConfig.getConfig("addon_config")

            LightRagConfig(
                openai =
                    OpenAiConfig(
                        apiKey = System.getenv("OPENAI_API_KEY") ?: openaiConfig.getString("api_key"),
                        chatModelName = openaiConfig.getString("chat_model_name"),
                        embeddingModelName = openaiConfig.getString("embedding_model_name"),
                        embeddingModelDimensions = openaiConfig.getInt("embedding_model_dimensions"),
                    ),
                ollama =
                    OllamaConfig(
                        baseUrl = System.getenv("OLLAMA_BASE_URL") ?: ollamaConfig.getString("base_url"),
                        chatModelName = ollamaConfig.getString("chat_model_name"),
                        embeddingModelName = ollamaConfig.getString("embedding_model_name"),
                    ),
                neo4j =
                    Neo4jConfig(
                        uri = System.getenv("NEO4J_URI") ?: neo4jConfig.getString("uri"),
                        username = System.getenv("NEO4J_USERNAME") ?: neo4jConfig.getString("username"),
                        password = System.getenv("NEO4J_PASSWORD") ?: neo4jConfig.getString("password"),
                    ),
                mongodb =
                    MongoDbConfig(
                        uri = System.getenv("MONGO_URI") ?: mongodbConfig.getString("uri"),
                        database = System.getenv("MONGO_DB") ?: mongodbConfig.getString("database"),
                    ),
                storage =
                    StorageConfig(
                        workingDir = storageConfig.getString("working_dir"),
                        graphStorageName = storageConfig.getString("graph_storage_name"),
                        vectorStorageName = storageConfig.getString("vector_storage_name"),
                    ),
                addonConfig =
                    AddonConfigConfig(
                        chunkTokenSize = addonConfigValues.getInt("chunk_token_size"),
                        chunkOverlapTokenSize = addonConfigValues.getInt("chunk_overlap_token_size"),
                        cosineBetterThreshold = addonConfigValues.getDouble("cosine_better_threshold"),
                        entityTypes = addonConfigValues.getStringList("entity_types"),
                        language = addonConfigValues.getString("language"),
                    ),
                resetStorage =
                    System.getenv("LIGHTRAG_RESET_STORAGE")?.toBoolean()
                        ?: lightragConfig.getBoolean("reset_storage"),
            )
        }

        single {
            val lightRagConfig = get<LightRagConfig>()
            val chatModel =
                LLMFactory.createChatModel(
                    binding = "openai",
                    modelName = lightRagConfig.openai.chatModelName,
                    apiKey = lightRagConfig.openai.apiKey,
                )
            val streamingChatModel =
                LLMFactory.createStreamingChatModel(
                    binding = "openai",
                    modelName = lightRagConfig.openai.chatModelName,
                    apiKey = lightRagConfig.openai.apiKey,
                )
            DualChatModel(chatModel, streamingChatModel)
        }

        single<ChatModel> { get<DualChatModel>() }

        single<StreamingChatModel> { get<DualChatModel>() }

        single<EmbeddingModel> {
            val lightRagConfig = get<LightRagConfig>()
            LLMFactory.createEmbeddingModel(
                binding = "openai",
                modelName = lightRagConfig.openai.embeddingModelName,
                apiKey = lightRagConfig.openai.apiKey,
            )
        }

        single {
            val lightRagConfig = get<LightRagConfig>()
            AppConfig(
                chatModel = get(),
                embeddingModel = get(),
                workingDir = lightRagConfig.storage.workingDir,
                graphStorageName = lightRagConfig.storage.graphStorageName,
                vectorStorageName = lightRagConfig.storage.vectorStorageName,
                addonConfig =
                    AddonConfig(
                        neo4j = lightRagConfig.neo4j,
                        overrides =
                            LightRagOverrides(
                                chunkTokenSize = lightRagConfig.addonConfig.chunkTokenSize,
                                chunkOverlapTokenSize = lightRagConfig.addonConfig.chunkOverlapTokenSize,
                                entityTypes = lightRagConfig.addonConfig.entityTypes,
                                language = lightRagConfig.addonConfig.language,
                                cosineBetterThreshold = lightRagConfig.addonConfig.cosineBetterThreshold,
                            ),
                        cosineBetterThreshold = lightRagConfig.addonConfig.cosineBetterThreshold,
                    ),
            )
        }

        single { Encodings.newDefaultEncodingRegistry() }
        single { get<EncodingRegistry>().getEncoding(EncodingType.CL100K_BASE) }

        single(named("tokenizer")) {
            val enc = get<Encoding>()
            val tokenizer: (String) -> List<Int> = { text: String ->
                val intArrayList = enc.encode(text)
                val list = mutableListOf<Int>()
                for (i in 0 until intArrayList.size()) {
                    list.add(intArrayList.get(i))
                }
                list
            }
            tokenizer
        }

        single(named("decoder")) {
            val enc = get<Encoding>()
            val decoder: (List<Int>) -> String = { list ->
                val intArrayList =
                    com.knuddels.jtokkit.api
                        .IntArrayList()
                list.forEach { intArrayList.add(it) }
                enc.decode(intArrayList)
            }
            decoder
        }

        single(named("globalConfig")) {
            val appConfig = get<AppConfig>()
            val overrides = appConfig.addonConfig.overrides
            val chunkTokenSize = overrides.chunkTokenSize ?: 1200
            val chunkOverlapTokenSize = overrides.chunkOverlapTokenSize ?: 100
            val entityTypes = overrides.entityTypes ?: listOf("Person", "Organization", "Location", "Event", "Concept")
            val language = overrides.language ?: "English"
            mapOf(
                "llm_model_func" to appConfig.chatModel,
                "embedding_func" to appConfig.embeddingModel,
                "neo4j" to get<LightRagConfig>().neo4j,
                "chunk_token_size" to chunkTokenSize,
                "chunk_overlap_token_size" to chunkOverlapTokenSize,
                "entity_types" to entityTypes,
                "language" to language,
                "working_dir" to appConfig.workingDir,
                "enable_llm_cache" to (appConfig.hashingKv != null),
            ) + appConfig.addonConfig.toMap()
        }

        single {
            val appConfig = get<AppConfig>()
            StorageManager(
                workingDir = appConfig.workingDir,
                embeddingModel = appConfig.embeddingModel,
                graphStorageName = appConfig.graphStorageName,
                vectorStorageName = appConfig.vectorStorageName,
                addonConfig = appConfig.addonConfig,
                globalConfig = get(named("globalConfig")),
                docStatusStorageOverride = appConfig.docStatusStorageOverride,
                fullDocsStorageOverride = appConfig.fullDocsStorageOverride,
                textChunksStorageOverride = appConfig.textChunksStorageOverride,
                fullEntitiesStorageOverride = appConfig.fullEntitiesStorageOverride,
                fullRelationsStorageOverride = appConfig.fullRelationsStorageOverride,
            )
        }

        single {
            IngestionService(
                storageManager = get(),
                globalConfig = get(named("globalConfig")),
                tokenizer = get(named("tokenizer")),
                decoder = get(named("decoder")),
            )
        }

        single {
            val appConfig = get<AppConfig>()
            QueryService(
                storageManager = get(),
                chatModel = get(),
                hashingKv = appConfig.hashingKv,
                globalConfig = get(named("globalConfig")),
                tokenizer = get(named("tokenizer")),
                decoder = get(named("decoder")),
            )
        }

        single {
            LightRAG(
                ingestionService = get(),
                queryService = get(),
                storageManager = get(),
            )
        }
    }
