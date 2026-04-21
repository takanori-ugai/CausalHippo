java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.OpenAlexIntroMusiQueRagas --input data/openalex_eval_dataset_intro_musique.jsonl > logs/intro.log
java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.OpenAlexIntroMusiQueRagas --input data/openalex_eval_dataset_musique.jsonl > logs/abstract.log
java -cp build/libs/causalrag-0.0.1-all.jar causalrag.examples.OpenAlexIntroMusiQueRagas --input data/openalex_eval_dataset_fulltext_musique.jsonl > logs/fulltext.log
