java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.MusiQueRagas --input data/Data-Common_Sense_Causation.jsonl > logs/CausalHippo-CausalQA.log
java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.MusiQueCausalRagas --input data/Data-Common_Sense_Causation.jsonl > logs/Causal-CausalQA.log
java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.MusiQueStandardRagas --input data/Data-Common_Sense_Causation.jsonl > logs/Standard-CausalQA.log
java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.MusiQueHippoRagas --input data/Data-Common_Sense_Causation.jsonl > logs/Hippo-CausalQA.log
