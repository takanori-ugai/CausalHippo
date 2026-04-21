# Prompt Subsystem

## Main components

- `SimplePrompt`
- `TypedPrompt` / `BasePrompt`
- `ImageTextTypedPrompt`
- `DynamicFewShotPrompt`
- `PromptCollection`
- `PromptContentPart`

## Capabilities

- Prompt formatting with instruction/examples/input-output framing
- Structured JSON output parsing and retry correction
- Prompt save/load with stable hashing
- Optional example selection using embedding similarity (`DynamicFewShotPrompt`)
- Multimodal prompt content (`Text`, `ImageDataUri`, `ImageUrl`)
