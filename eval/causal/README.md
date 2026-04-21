# Causal Relation Benchmark Data Generator

This project is a Kotlin-based application designed to generate a synthetic dataset for benchmarking causal analysis models. It leverages the OpenAI GPT-4 model via the LangChain4j library to create complex text samples where the causal relationship between two entities is subtle and requires counterfactual reasoning to identify correctly.

The generated paragraphs are specifically crafted to suggest a plausible but potentially deceptive correlation, making the dataset challenging and valuable for evaluating advanced NLP models.

## Features

- **High-Quality Data Generation**: Uses OpenAI's `gpt-4-0125-preview` to generate nuanced and context-rich text.
- **Causal Inference Focus**: Creates examples specifically for the "Counterfactual Reasoning Required" category of causal analysis.
- **Structured Output**: Produces a clean, ready-to-use CSV file (`.csv`).
- **Powered by LangChain4j**: Integrates seamlessly with large language models (LLMs) using the powerful and easy-to-use LangChain4j framework.

## Prerequisites

- **JDK 11** or later
- **Gradle** or **Maven**
- **OpenAI API Key**

## Setup and Configuration

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd CausalRelationBenchmark
    ```

2.  **Configure the OpenAI API Key:**

    For security reasons, you should not hardcode your API key directly in the source code. It is highly recommended to use environment variables.

    **a. Set an Environment Variable:**

    -   **Linux/macOS:**
        ```bash
        export OPENAI_API_KEY="your_secret_key_here"
        ```
    -   **Windows (PowerShell):**
        ```powershell
        $env:OPENAI_API_KEY="your_secret_key_here"
        ```

    **b. Update the Code:**

    Modify `src/main/kotlin/jp/live/ugai/Main.kt` to read the key from the environment variable instead of the hardcoded constant.

    ```kotlin
    // In Main.kt

    // private const val API_KEY = "sk-..." // REMOVE THIS LINE
    private val API_KEY = System.getenv("OPENAI_API_KEY")
    private const val TEMPERATURE = 0.8
    
    fun main() {
        if (API_KEY.isNullOrBlank()) {
            println("Error: OPENAI_API_KEY environment variable not set.")
            return
        }
        // ... rest of the main function
    }
    ```

3.  **Build the project:**

    Use your build tool to download dependencies and compile the code.
    ```bash
    # Using Gradle
    ./gradlew build
    ```

## Usage

Run the `main` function from your IDE or via the command line. The application will start sending prompts to the OpenAI API.

```bash
# Using Gradle
./gradlew run
```

The script will execute a loop 20 times. In each iteration, it generates 25 new text samples and appends them to the output file. A progress indicator (`0`, `1`, `2`, ...) will be printed to the console for each completed iteration.

### Output

The generated data will be appended to a file named `Data-Counterfactual_Reasoning_Required.csv` in the project's root directory.

The CSV file has the following columns:

- `e1`: The first entity (potential cause).
- `e2`: The second entity (potential effect).
- `text`: The full paragraph containing the entities, marked with `<e1>` and `<e2>` tags.
- `answer`: A boolean (`true` if `e1` causes `e2`, `false` otherwise).
- `category`: The classification of the causal relationship (e.g., "Counterfactual Reasoning Required").