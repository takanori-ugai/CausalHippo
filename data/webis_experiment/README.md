# Webis Experiment Data Summary

Total rows converted: 868281
Lexical overlap tertiles: q1=0.054545, q2=0.080000

## Source counts
- gooaq: 146253 (context-missing fallback: 146253)
- hotpotqa: 355 (context-missing fallback: 0)
- msmarco: 23011 (context-missing fallback: 0)
- naturalquestions: 1137 (context-missing fallback: 0)
- newsqa: 623 (context-missing fallback: 0)
- paq: 692645 (context-missing fallback: 0)
- searchqa: 663 (context-missing fallback: 0)
- squad2: 2957 (context-missing fallback: 0)
- triviaqa: 637 (context-missing fallback: 8)

## Balanced subset quotas
- gooaq: 34
- hotpotqa: 34
- msmarco: 34
- naturalquestions: 33
- newsqa: 33
- paq: 33
- searchqa: 33
- squad2: 33
- triviaqa: 33

## Overlap bucket counts
- low: 294926
- mid: 285198
- high: 288157

## Configuration
- balanced_size: 300
- max_words_per_paragraph: 150
- hard_negatives_per_sample: 3
- hard_negative_pool_size: 20000
- write_all: False
- write_source_splits: False
- write_overlap_splits: False

## Generated files
- webis_train_balanced_300.jsonl: 300 rows
- webis_train_balanced_300_hardneg3.jsonl: 300 rows
- webis_train_manifest.jsonl: 868281 rows
