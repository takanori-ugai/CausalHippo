# Graph

## Core types

- `Node`
- `Relationship`
- `KnowledgeGraph`
- `NodeType` (`UNKNOWN`, `DOCUMENT`, `CHUNK`)

## Helpers

- `getChildNodes(node, graph, level)`
- `getParentNodes(node, graph, level)`

`KnowledgeGraph` supports JSON save/load via `save(path)` and `load(path)`.
