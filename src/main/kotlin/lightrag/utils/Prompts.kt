package lightrag.utils

/**
 * A collection of prompts used throughout the LightRAG application.
 */
object Prompts {
    /**
     * The system prompt for entity extraction.
     */
    const val ENTITY_EXTRACTION_SYSTEM_PROMPT = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and relationships from the input text.

---Instructions---
1.  **Entity and Relationship Extraction:**
    *   Identify meaningful entities and the relationships between them from the input text.
    *   Ensure consistent naming for entities across the entire extraction process.

2.  **Output Format:**
    *   Each extracted entity should be on a new line, starting with `entity<|#|>` followed by `name<|#|>type<|#|>description`.
    *   Each extracted relationship should be on a new line, starting with `relation<|#|>` followed by `source<|#|>target<|#|>keywords<|#|>description`.
    *   `name`: The name of the entity (string).
    *   `type`: The type of the entity from the list: `{entity_types}` (string).
    *   `description`: A concise description of the entity based on the text (string).
    *   `source`: The name of the source entity (string).
    *   `target`: The name of the target entity (string).
    *   `keywords`: High-level keywords summarizing the relationship, separated by commas (string).
    *   `description`: A concise explanation of the relationship (string).

3.  **Content Guidelines:**
    *   All extracted information must be based solely on the provided input text.
    *   Write all names and descriptions in the third person and in {language}.
    *   Avoid using pronouns like `this article`, `I`, `you`.
    *   Retain proper nouns in their original language.

"""

    /**
     * The user prompt for entity extraction.
     */
    const val ENTITY_EXTRACTION_USER_PROMPT = """---Task---
Extract entities and relationships from the input text in Data to be Processed below.

---Instructions---
1.  **Strict Adherence to Format:** Produce extracted `entities` and `relations` as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted entities and relations in the specified format. Do not include any introductory or concluding remarks, explanations, or additional text before or after the extracted data.
3.  **Output Language:** Ensure the output language is {language}.

---Data to be Processed---
<Entity_types>
[{entity_types}]

<Input Text>
```
{input_text}
```

<Output>
"""

    /**
     * Examples for entity extraction.
     */
    val ENTITY_EXTRACTION_EXAMPLES =
        listOf(
            """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
```

<Output>
```json
{
  "entities": [
    {
      "name": "Alex",
      "type": "person",
      "description": "Alex is a character who experiences frustration and is observant of the dynamics among other characters."
    },
    {
      "name": "Taylor",
      "type": "person",
      "description": "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."
    },
    {
      "name": "Jordan",
      "type": "person",
      "description": "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."
    },
    {
      "name": "Cruz",
      "type": "person",
      "description": "Cruz is associated with a vision of control and order, influencing the dynamics among other characters."
    },
    {
      "name": "The Device",
      "type": "equipment",
      "description": "The Device is central to the story, with potential game-changing implications, and is revered by Taylor."
    }
  ],
  "relations": [
    {
      "source": "Alex",
      "target": "Taylor",
      "keywords": "power dynamics, observation",
      "description": "Alex observes Taylor's authoritarian behavior and notes changes in Taylor's attitude toward the device."
    },
    {
      "source": "Alex",
      "target": "Jordan",
      "keywords": "shared goals, rebellion",
      "description": "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."
    },
    {
      "source": "Taylor",
      "target": "Jordan",
      "keywords": "conflict resolution, mutual respect",
      "description": "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."
    },
    {
      "source": "Jordan",
      "target": "Cruz",
      "keywords": "ideological conflict, rebellion",
      "description": "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."
    },
    {
      "source": "Taylor",
      "target": "The Device",
      "keywords": "reverence, technological significance",
      "description": "Taylor shows reverence towards the device, indicating its importance and potential impact."
    }
  ]
}
```
""",
            """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the global tech index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, nexon technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching ${'$'}2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to ${'$'}87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.
```

<Output>
```json
{
  "entities": [
    {
      "name": "Global Tech Index",
      "type": "category",
      "description": "The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today."
    },
    {
      "name": "Nexon Technologies",
      "type": "organization",
      "description": "Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings."
    },
    {
      "name": "Omega Energy",
      "type": "organization",
      "description": "Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices."
    },
    {
      "name": "Gold Futures",
      "type": "product",
      "description": "Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets."
    },
    {
      "name": "Crude Oil",
      "type": "product",
      "description": "Crude oil prices rose to ${'$'}87.60 per barrel due to supply constraints and strong demand."
    },
    {
      "name": "Market Selloff",
      "type": "category",
      "description": "Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations."
    },
    {
      "name": "Federal Reserve Policy Announcement",
      "type": "category",
      "description": "The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability."
    },
    {
      "name": "3.4% Decline",
      "type": "category",
      "description": "The Global Tech Index experienced a 3.4% decline in midday trading."
    }
  ],
  "relations": [
    {
      "source": "Global Tech Index",
      "target": "Market Selloff",
      "keywords": "market performance, investor sentiment",
      "description": "The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns."
    },
    {
      "source": "Nexon Technologies",
      "target": "Global Tech Index",
      "keywords": "company impact, index movement",
      "description": "Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index."
    },
    {
      "source": "Gold Futures",
      "target": "Market Selloff",
      "keywords": "market reaction, safe-haven investment",
      "description": "Gold prices rose as investors sought safe-haven assets during the market selloff."
    },
    {
      "source": "Federal Reserve Policy Announcement",
      "target": "Market Selloff",
      "keywords": "interest rate impact, financial regulation",
      "description": "Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff."
    }
  ]
}
```
""",
            """<Entity_types>
["Person","Creature","Organization","Location","Event","Concept","Method","Content","Data","Artifact","NaturalObject"]

<Input Text>
```
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
```

<Output>
```json
{
  "entities": [
    {
      "name": "World Athletics Championship",
      "type": "event",
      "description": "The World Athletics Championship is a global sports competition featuring top athletes in track and field."
    },
    {
      "name": "Tokyo",
      "type": "location",
      "description": "Tokyo is the host city of the World Athletics Championship."
    },
    {
      "name": "Noah Carter",
      "type": "person",
      "description": "Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship."
    },
    {
      "name": "100m Sprint Record",
      "type": "category",
      "description": "The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter."
    },
    {
      "name": "Carbon-Fiber Spikes",
      "type": "equipment",
      "description": "Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction."
    },
    {
      "name": "World Athletics Federation",
      "type": "organization",
      "description": "The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations."
    }
  ],
  "relations": [
    {
      "source": "World Athletics Championship",
      "target": "Tokyo",
      "keywords": "event location, international competition",
      "description": "The World Athletics Championship is being hosted in Tokyo."
    },
    {
      "source": "Noah Carter",
      "target": "100m Sprint Record",
      "keywords": "athlete achievement, record-breaking",
      "description": "Noah Carter set a new 100m sprint record at the championship."
    },
    {
      "source": "Noah Carter",
      "target": "Carbon-Fiber Spikes",
      "keywords": "athletic equipment, performance boost",
      "description": "Noah Carter used carbon-fiber spikes to enhance performance during the race."
    },
    {
      "source": "Noah Carter",
      "target": "World Athletics Championship",
      "keywords": "athlete participation, competition",
      "description": "Noah Carter is competing at the World Athletics Championship."
    }
  ]
}
```
""",
        )

    /**
     * The prompt for summarizing entity descriptions.
     */
    const val SUMMARIZE_ENTITY_DESCRIPTIONS = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""

    /**
     * The response to return when no context is found.
     */
    const val FAIL_RESPONSE = "Sorry, I'm not able to provide an answer to that question.[no-context]"

    /**
     * The prompt for generating a RAG response.
     */
    const val RAG_RESPONSE = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

    /**
     * The prompt for generating a naive RAG response.
     */
    const val NAIVE_RAG_RESPONSE = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

    /**
     * The context for a knowledge graph query.
     */
    const val KG_QUERY_CONTEXT = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

    /**
     * The context for a naive query.
     */
    const val NAIVE_QUERY_CONTEXT = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

    /**
     * The prompt for keywords extraction.
     */
    const val KEYWORDS_EXTRACTION = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.
5. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

    /**
     * Examples for keywords extraction.
     */
    val KEYWORDS_EXTRACTION_EXAMPLES =
        listOf(
            """Example 1:

Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}
""",
            """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}
""",
            """Example 3:

Query: "What is the role of education in reducing poverty?"

Output:
{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}
""",
        )
}
