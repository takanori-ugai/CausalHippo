package shared.rag.unified;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal usage sample for the unified API (Java version).
 */
public final class UnifiedSampleJava {
    private UnifiedSampleJava() {}

    public static void main(String[] args) {
        String question = "How does Alpha relate to Gamma?";
        String openAiApiKey = System.getenv("OPENAI_API_KEY");
        if (openAiApiKey == null || openAiApiKey.isBlank()) {
            throw new IllegalStateException("OPENAI_API_KEY is required to run this sample with OpenAI embeddings.");
        }

        Map<String, Object> overrides = new LinkedHashMap<>();
        overrides.put("useUnifiedPersistence", true);
        overrides.put("persistenceBackend", "filesystem_snapshot");
        overrides.put("useUnifiedSpiForRetrievalAndIndex", true);
        overrides.put("embeddingApiKey", openAiApiKey);
        overrides.put("embeddingModel", "text-embedding-3-small");

        UnifiedRagHandle handle = UnifiedRagFactory.INSTANCE.create(RagId.CAUSAL_RAG, null, overrides);
        try {
            List<String> passages = Arrays.asList("Alpha influences Beta.", "Beta affects Gamma.");
            handle.getRag().upsert(passages);

            UnifiedQuery query =
                    new UnifiedQuery(
                            "",
                            UnifiedMode.HYBRID,
                            5,
                            true,
                            true,
                            true,
                            false,
                            false,
                            false,
                            null,
                            Collections.emptyList(),
                            Collections.emptyMap());

            UnifiedResponse response = handle.getRag().query(question, query);

            System.out.println("RAG: " + handle.getId());
            System.out.println(
                    "Persistence backend: "
                            + (handle.getPersistence() != null ? handle.getPersistence().getBackendId() : "none"));
            System.out.println("Answer: " + (response.getAnswer() != null ? response.getAnswer() : "(no answer)"));
            System.out.println("Context rows: " + response.getContext().size());
        } finally {
            handle.close();
        }
    }
}
