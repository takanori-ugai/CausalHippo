package ragas.backends

/**
 * Contract for experiment/dataset persistence backends.
 */
interface BaseBackend {
    /**
     * Loads a dataset by name from the backend.
     *
     * @param name Name or identifier.
     * @return List of rows for the dataset, where each map is one row of column values.
     */
    fun loadDataset(name: String): List<Map<String, Any?>>

    /**
     * Loads an experiment by name from the backend.
     *
     * @param name Name or identifier.
     * @return List of rows for the experiment, where each map is one row of column values.
     */
    fun loadExperiment(name: String): List<Map<String, Any?>>

    /**
     * Persists dataset rows under the provided dataset name.
     *
     * @param name Name or identifier.
     * @param data Rows to persist for the dataset, where each map is one row.
     */
    fun saveDataset(
        name: String,
        data: List<Map<String, Any?>>,
    )

    /**
     * Persists experiment rows under the provided experiment name.
     *
     * @param name Name or identifier.
     * @param data Rows to persist for the experiment, where each map is one row.
     */
    fun saveExperiment(
        name: String,
        data: List<Map<String, Any?>>,
    )

    /**
     * Lists available dataset names.
     */
    fun listDatasets(): List<String>

    /**
     * Lists available experiment names.
     */
    fun listExperiments(): List<String>
}
