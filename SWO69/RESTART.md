  ### 3. Recommended Action Before Running

  Because E0 LightRAG ran before the fix and did not save its kg_snapshot inside the e0_lightrag_musique directory,
  delete its per_question output so it regenerates its snapshot (takes ~7s):

    # 1. Remove the old E0 LightRAG marker so its snapshot is regenerated
    rm -f eval_results/swo69_20260819_043352/e0_lightrag_musique/per_question/lightrag.jsonl

    # 2. Remove the premature DONE line from run.log (if running mode=all)
    sed -i '/DONE mode=all/d' eval_results/swo69_20260819_043352/logs/run.log

    # 3. Launch continuation supervisor
    bash SWO69/continue_swo69.sh run
