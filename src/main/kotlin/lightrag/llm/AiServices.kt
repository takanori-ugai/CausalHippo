package lightrag.llm

import dev.langchain4j.service.SystemMessage
import dev.langchain4j.service.UserMessage
import dev.langchain4j.service.V
import lightrag.operate.ExtractionResult
import lightrag.operate.KeywordsExtractionResult

/**
 * An AI service for extracting entities and relationships from text.
 */
interface EntityExtractor {
    /**
     * Extracts entities and relationships from the given text.
     * @param text The text to extract from.
     * @param entityTypes The types of entities to extract.
     * @param language The language of the text.
     * @return An [ExtractionResult] containing the extracted entities and relationships.
     */
    @SystemMessage(
        """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity and Relationship Extraction:**
    *   Identify meaningful entities and the relationships between them from the input text.
    *   Ensure consistent naming for entities across the entire extraction process.

2.  **Output Format:**
    *   The output must be a single valid JSON object.
    *   The JSON object should have two top-level keys: `entities` and `relations`.
    *   `entities`: An array of objects, where each object represents an entity and has the following keys:
        *   `name`: The name of the entity (string).
        *   `type`: The type of the entity from the list: `{{entity_types}}` (string).
        *   `description`: A concise description of the entity based on the text (string).
    *   `relations`: An array of objects, where each object represents a relationship and has the following keys:
        *   `source`: The name of the source entity (string).
        *   `target`: The name of the target entity (string).
        *   `keywords`: High-level keywords summarizing the relationship, separated by commas (string).
        *   `description`: A concise explanation of the relationship (string).

3.  **Content Guidelines:**
    *   All extracted information must be based solely on the provided input text.
    *   Write all names and descriptions in the third person and in {{language}}.
    *   Avoid using pronouns like `this article`, `I`, `you`.
    *   Retain proper nouns in their original language.

---JSON Schema Example---
```json
{
  "entities": [
    {
      "name": "Entity Name",
      "type": "Person",
      "description": "Description of the entity."
    }
  ],
  "relations": [
    {
      "source": "Entity Name",
      "target": "Another Entity Name",
      "keywords": "keyword1, keyword2",
      "description": "Description of the relationship."
    }
    ]
}
```
""",
    )
    fun extract(
        @V("input_text")
        @UserMessage(
            """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Produce a single, valid JSON object containing the extracted `entities` and `relations` as specified in the system prompt.
2.  **Output Content Only:** Output *only* the JSON object. Do not include any introductory or concluding remarks, explanations, or additional text before or after the JSON.
3.  **Output Language:** Ensure the output language is {{language}}.

---Data to be Processed---
<Entity_types>
[{{entity_types}}]

<Input Text>
```
{{input_text}}
```

<Output>
""",
        )
        text: String,
        @V("entity_types") entityTypes: String,
        @V("language") language: String,
    ): ExtractionResult
}

/**
 * An AI service for extracting keywords from text.
 */
interface KeywordExtractor {
    /**
     * Extracts keywords from the given text.
     * @param text The text to extract from.
     * @param language The language of the text.
     * @param examples Examples of keywords to extract.
     * @return A [KeywordsExtractionResult] containing the extracted keywords.
     */
    @SystemMessage(
        """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any introductory or concluding remarks, explanations, or additional text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {{language}}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{{examples}}

---Real Data---
User Query: {{query}}

---Output---
Output:""",
    )
    fun extract(
        @V("query")
        @UserMessage(
            """User Query: {{query}}""",
        )
        text: String,
        @V("language") language: String,
        @V("examples") examples: String,
    ): KeywordsExtractionResult
}
