package ragas.backends

import java.util.concurrent.ConcurrentHashMap

/**
 * In-memory backend implementation for datasets and experiments, intended for tests and local runs.
 */
class InMemoryBackend : BaseBackend {
    private val datasets = ConcurrentHashMap<String, List<Map<String, Any?>>>()
    private val experiments = ConcurrentHashMap<String, List<Map<String, Any?>>>()

    /**
     * Loads a dataset by name from the backend.
     */
    override fun loadDataset(name: String): List<Map<String, Any?>> =
        datasets[name]?.map { row -> row.toMap() }
            ?: throw java.io.FileNotFoundException("Dataset '$name' not found")

    /**
     * Loads an experiment by name from the backend.
     */
    override fun loadExperiment(name: String): List<Map<String, Any?>> =
        experiments[name]?.map { row -> row.toMap() }
            ?: throw java.io.FileNotFoundException("Experiment '$name' not found")

    /**
     * Persists dataset rows under the provided dataset name.
     */
    override fun saveDataset(
        name: String,
        data: List<Map<String, Any?>>,
    ) {
        datasets[name] = data.map { row -> row.toMap() }
    }

    /**
     * Persists experiment rows under the provided experiment name.
     */
    override fun saveExperiment(
        name: String,
        data: List<Map<String, Any?>>,
    ) {
        experiments[name] = data.map { row -> row.toMap() }
    }

    /**
     * Lists available dataset names.
     */
    override fun listDatasets(): List<String> = datasets.keys.sorted()

    /**
     * Lists available experiment names.
     */
    override fun listExperiments(): List<String> = experiments.keys.sorted()
}
