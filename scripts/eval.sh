java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.MusiQueCausalRagas --input data/musique_ans_v1.0_train-200.jsonl > logs/Causal.log
java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.MusiQueStandardRagas --input data/musique_ans_v1.0_train-200.jsonl > logs/Standard.log
java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.MusiQueHippoRagas --input data/musique_ans_v1.0_train-200.jsonl > logs/Hippo.log
