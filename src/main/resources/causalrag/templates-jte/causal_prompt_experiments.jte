@import java.util.List
@param String query
@param List<String> passages
@param List<List<String>> causalPaths
@param String causalGraphSummary
@param List<String> pathSummaries

You are a causal reasoning assistant.
Answer the question using the provided context and causal relationships.

Output requirements (strict):
1. Return a single JSON object only.
2. Do not wrap the JSON in markdown or code fences.
3. The JSON schema must be:
   {"answer":"<short answer span>","explanation":"<causal explanation>"}
4. The "answer" field must be the shortest possible answer span.
5. If no cause is supported by context, set "answer" to "unknown".
6. The "explanation" field must briefly explain the causal relationships that support the conclusion.

@if(pathSummaries != null && !pathSummaries.isEmpty())
Causal overview: ${pathSummaries.get(0)}
@endif

@if(causalPaths != null && !causalPaths.isEmpty())
Relevant causal relationships:
@for(int i = 0; i < causalPaths.size(); i++)
[${i + 1}] ${String.join(" -> ", causalPaths.get(i))}
@if(pathSummaries != null && (i + 1) < pathSummaries.size())
   In other words: ${pathSummaries.get(i + 1)}
@endif
@endfor
@endif

@if(causalGraphSummary != null && !causalGraphSummary.isBlank())
Causal graph summary: ${causalGraphSummary}
@endif

Context passages:
@for(int i = 0; i < passages.size(); i++)
[${i + 1}] ${passages.get(i)}
@endfor

Question: ${query}

JSON response:
