#!/usr/bin/env bash
# ============================================================================
# SWO69 batch-job continuation supervisor — SWO_PLAN.md §5 / §7-T6 / §8
#
# The full experiment takes DAYS, but rt_HG allocations are capped at ~3 h
# per job and expiry SIGKILLs the whole process tree (dominant failure mode
# since 2026-08-19). This supervisor turns a chain of such jobs into ONE
# logical batch job:
#
#   * (Re)starts `run_experiment_swo69.sh <mode>` in the fixed run dir.
#     Resume-safe: conditions whose per_question/<cond>.jsonl already has
#     rows are skipped by the run script itself.
#   * Queues the NEXT job (this same script) EARLY, so chain liveness never
#     depends on this allocation surviving.
#   * Watches the run; if the run script dies mid-condition it is restarted
#     (bounded per session and globally; the dying condition's log tail is
#     captured into continue.log for diagnosis).
#   * Exits before walltime expires; the PBS kill takes the experiment down
#     with the tree; the successor — which first waits for us to terminate —
#     resumes from the last completed condition.
#   * Terminates the chain when `DONE mode=<mode>` reaches run.log (or when
#     the global restart cap is hit → chain halted for a human).
#
# Usage:
#   bash SWO69/continue_swo69.sh run      # supervisor loop (default)
#   bash SWO69/continue_swo69.sh status   # read-only state report
#   bash SWO69/continue_swo69.sh submit   # queue a successor now, exit
#
# Environment overrides:
#   SVO69_TS               run stamp (default: 20260819_043352 — the live dir)
#   SVO69_MODE             all|e0|e1|e2|analyze (default: all)
#   SVO69_PREDECESSOR      PBS job id of the previous chain job (auto-filled)
#   SVO69_RESERVE          exit this many seconds before walltime (900)
#   SVO69_POLL             watch-loop interval seconds (60)
#   SVO69_SESSION_RESTARTS max run-script restarts per supervisor session (3)
#   SVO69_GLOBAL_RESTARTS  max restarts total; exceeded → chain halts (8)
#   SVO69_QUEUE            PBS queue (default rt_HG)
#   SVO69_PROJECT          PBS project (default gah51681)
#   SVO69_WALLTIME         requested walltime (12:00:00; the queue caps it)
#   SVO69_JOBNAME          chain job name (default swo69_batch)
#
# State (under eval_results/swo69_<ts>/logs/):
#   continue.log     supervisor event log (appended across jobs)
#   chain_state      predecessor= / successor= / attempts=  (attempts is the
#                     global restart counter; `grep -v '^attempts=' chain_state`
#                     or delete the line to reset it after a fix)
#   swo69.lock.d/    exclusive lock (atomic mkdir + holder pid; stale locks
#                    from dead supervisors are reclaimed) — one supervisor per
#                    run dir. NOT flock: a lock fd would be inherited by the
#                    long-lived run-script child and outlive the supervisor.
#   run_launcher.pid pid of the run script this session started
#
# NOTE: per_question rows are written ONCE per condition at its end (Kotlin
# writeJsonl), so a kill mid-condition leaves no partial file and the
# condition simply re-runs — never a false "already done" skip.
# ============================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="${SVO69_TS:-20260819_043352}"
MODE="${SVO69_MODE:-all}"
RUN_DIR="$REPO/eval_results/swo69_${TS}"
LOGS="$RUN_DIR/logs"
RUN_LOG="$LOGS/run.log"
CONT_LOG="$LOGS/continue.log"
CHAIN_STATE="$LOGS/chain_state"
LOCK="$LOGS/swo69.lock"

QUEUE="${SVO69_QUEUE:-rt_HG}"
PROJECT="${SVO69_PROJECT:-gah51681}"
WALLTIME="${SVO69_WALLTIME:-12:00:00}"
JOBNAME="${SVO69_JOBNAME:-swo69_batch}"
RESERVE="${SVO69_RESERVE:-900}"
POLL="${SVO69_POLL:-60}"
SESSION_MAX="${SVO69_SESSION_RESTARTS:-3}"
GLOBAL_MAX="${SVO69_GLOBAL_RESTARTS:-8}"
JDK_HOME="${JDK_HOME:-$HOME/jdk-21.0.6}"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/bin/ollama}"
OLLAMA_URL="http://127.0.0.1:11434"

mkdir -p "$LOGS"

log() { echo "=== [swo69-continue] $(date -u +%FT%TZ) $* ===" | tee -a "$CONT_LOG"; }

# --- lock: atomic mkdir + holder pid (no fd to leak into child processes) ---
LOCKDIR="${LOCK}.d"
lock_acquire() {
  if mkdir "$LOCKDIR" 2>/dev/null; then
    echo $$ > "$LOCKDIR/pid"
    return 0
  fi
  local holder
  holder=$(cat "$LOCKDIR/pid" 2>/dev/null)
  if [[ -n "$holder" ]] && kill -0 "$holder" 2>/dev/null; then
    echo "another supervisor (pid $holder) holds $LOCKDIR — exiting" >&2
    return 1
  fi
  # Stale lock from a dead supervisor — reclaim.
  rm -rf "$LOCKDIR"
  if ! mkdir "$LOCKDIR" 2>/dev/null; then
    echo "lock contention on $LOCKDIR — exiting" >&2
    return 1
  fi
  echo $$ > "$LOCKDIR/pid"
  return 0
}

# --- chain state helpers ----------------------------------------------------
chain_get() { # key -> value (empty if unset)
  [[ -f "$CHAIN_STATE" ]] || return 0
  sed -n "s/^$1=//p" "$CHAIN_STATE" | tail -1
}
chain_set() { # key value
  local k="$1" v="$2"
  if [[ -f "$CHAIN_STATE" ]]; then
    local tmp; tmp=$(mktemp)
    grep -v "^${k}=" "$CHAIN_STATE" > "$tmp" 2>/dev/null || true
    echo "${k}=${v}" >> "$tmp"
    mv "$tmp" "$CHAIN_STATE"
  else
    echo "${k}=${v}" > "$CHAIN_STATE"
  fi
}

# --- status probes ----------------------------------------------------------
is_done()   { [[ -f "$RUN_LOG" ]] && grep -qF "DONE mode=$MODE" "$RUN_LOG"; }
# Anchored: only a real "bash <path>/run_experiment_swo69.sh <mode>" process
# (as start_run launches it), NOT shells whose cmdline merely mentions the
# script name (e.g. a verifier running pgrep/grep with the name in its args).
run_alive() { pgrep -f '^bash [^ ]*run_experiment_swo69\.sh (all|e0|e1|e2|analyze)$' >/dev/null 2>&1; }

job_state() { # jobid -> single-letter state or empty (gone/unknown)
  qstat -f "$1" 2>/dev/null | sed -n 's/^[[:space:]]*job_state = //p' | tr -d ' '
}

chain_job_ids() { # ids of queued/running jobs named $JOBNAME
  qstat -f 2>/dev/null | awk -v n="$JOBNAME" '/^Job Id:/{id=$3} /Job_Name = /{if ($3==n) print id}'
}

walltime_remaining_s() { # -> seconds, or -1 when not inside a PBS job
  [[ -n "${PBS_JOBID:-}" ]] || { echo "-1"; return; }
  local f rl ru
  f=$(qstat -f "$PBS_JOBID" 2>/dev/null)
  rl=$(sed -n 's/^[[:space:]]*Resource_List\.walltime = //p' <<<"$f" | tr -d ' ')
  ru=$(sed -n 's/^[[:space:]]*resources_used\.walltime = //p' <<<"$f" | tr -d ' ')
  [[ -n "${rl:-}" && -n "${ru:-}" ]] || { echo "-1"; return; }
  # 10# on EVERY field: "09"/"08" seconds are invalid octal in bash arithmetic
  # (this silently emptied the output once and made the loop exit early).
  hms_to_s() { local a b c; IFS=: read -r a b c <<<"$1"; echo $(( 10#${a:-0}*3600 + 10#${b:-0}*60 + 10#${c:-0} )); }
  local out
  out=$(( $(hms_to_s "$rl") - $(hms_to_s "$ru") )) || { echo "-1"; return; }
  echo "$out"
}

# --- actions ----------------------------------------------------------------
ensure_ollama() {
  if curl -sf --max-time 5 "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
    log "Ollama already up"
    return 0
  fi
  log "Ollama not responding — starting: $OLLAMA_BIN serve"
  nohup "$OLLAMA_BIN" serve >> "$LOGS/ollama_serve.log" 2>&1 &
  local i
  for ((i = 0; i < 90; i++)); do
    sleep 2
    if curl -sf --max-time 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
      log "Ollama up"
      return 0
    fi
  done
  log "ERROR: Ollama did not come up (see $LOGS/ollama_serve.log)"
  return 1
}

start_run() {
  log "launching: bash run_experiment_swo69.sh $MODE (SVO69_TS=$TS)"
  nohup env SVO69_TS="$TS" bash "$REPO/run_experiment_swo69.sh" "$MODE" \
    >> "$LOGS/supervisor_launch.log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOGS/run_launcher.pid"
  sleep 5
  if kill -0 "$pid" 2>/dev/null; then
    log "run script alive (pid=$pid)"
    return 0
  fi
  log "ERROR: run script exited within 5 s of launch — tail of supervisor_launch.log:"
  tail -n 20 "$LOGS/supervisor_launch.log" 2>/dev/null | sed 's/^/    /' >> "$CONT_LOG"
  return 1
}

submit_successor() {
  local existing
  existing=$(chain_job_ids)
  if [[ -n "$existing" ]]; then
    log "a $JOBNAME job is already in the queue: $(echo $existing | tr '\n' ' ') — not duplicating"
    return 0
  fi
  local script="$LOGS/pbs_successor.sh"
  cat > "$script" <<EOF
#!/bin/sh
#PBS -N $JOBNAME
#PBS -q $QUEUE
#PBS -l select=1
#PBS -l walltime=$WALLTIME
#PBS -P $PROJECT
# Unique output per predecessor (this PBS does not expand %j)
#PBS -o $LOGS/pbs_from_${PBS_JOBID:-head}.out
PATH=/opt/pbs/bin:\$PATH
module load cuda/12.8
export JAVA_HOME=$JDK_HOME
export PATH=\$JAVA_HOME/bin:\$PATH
export GRADLE_USER_HOME=/tmp/ugai/gradle
export SVO69_TS=$TS
export SVO69_MODE=$MODE
export SVO69_PREDECESSOR=${PBS_JOBID:-none}
cd $REPO
exec bash SWO69/continue_swo69.sh run
EOF
  chmod +x "$script"
  local out jobid
  out=$(qsub "$script" 2>&1) || { log "ERROR: qsub failed: $out"; return 1; }
  jobid=$(sed -n 's/^\([0-9][0-9]*\)\.[A-Za-z0-9._-]*$/\1/p' <<<"$out" | head -1)
  if [[ -z "$jobid" ]]; then
    log "ERROR: qsub output not parseable: $out"
    return 1
  fi
  chain_set "successor" "$jobid"
  # File copy of the predecessor pointer (env is the primary channel).
  chain_set "predecessor" "${PBS_JOBID:-none}"
  log "successor queued: job=$jobid (it will wait for us to terminate, then resume)"
  return 0
}

qdel_chain_jobs() {
  local id
  for id in $(chain_job_ids); do
    if qdel "$id" 2>/dev/null; then log "qdel $id (chain halted)"; else log "WARN: qdel $id failed"; fi
  done
}

diagnose_and_restart() {
  local last name
  last=$(grep -F ' START ' "$RUN_LOG" 2>/dev/null | tail -1)
  name=$(sed -E 's/.*START ([a-z0-9_]+).*/\1/' <<<"${last:-}")
  log "RESTART: run script dead; last started condition: ${name:-unknown}"
  if [[ -n "$name" && -f "$LOGS/${name}.log" ]]; then
    {
      echo "    --- tail of ${name}.log ---"
      tail -n 15 "$LOGS/${name}.log" | sed 's/^/    /'
    } >> "$CONT_LOG"
  fi
  start_run || log "restart launch failed"
}

wait_predecessor() { # block until the previous chain job is gone, then grace
  local pred="$1" st
  st=$(job_state "$pred")
  if [[ -z "$st" || "$st" != "R" ]]; then
    log "predecessor ${pred} not running (state='${st:-gone}') — 30 s grace for tree teardown"
    sleep 30
    return 0
  fi
  log "predecessor ${pred} still running — waiting for it to be killed"
  while true; do
    sleep 30
    st=$(job_state "$pred")
    [[ -z "$st" || "$st" != "R" ]] && break
  done
  log "predecessor terminated (state='${st:-unknown}') — 60 s grace"
  sleep 60
}

# --- subcommands ------------------------------------------------------------
cmd_run() {
  # The queue has no job dependencies (qsub rejects #PBS -W dependafter) and
  # may start us BEFORE the predecessor job dies. A live predecessor
  # supervisor then holds the lock — so WAIT (cost: one idle slot) instead of
  # dying, which would cause submit/exit churn. The predecessor's own loop
  # keeps the torch alive either way.
  local waited=0 rem
  while ! lock_acquire; do
    if (( waited % 10 == 0 )); then
      log "waiting for the run-dir lock (a live supervisor holds it — predecessor still active)"
    fi
    waited=$((waited + 1))
    rem=$(walltime_remaining_s); case "$rem" in (''|*[!0-9]*) rem=-1 ;; esac
    if (( rem >= 0 && rem < 180 )); then
      log "our walltime nearly expired while waiting for the lock — exiting (the predecessor's chain continues)"
      exit 0
    fi
    sleep 30
  done
  log "supervisor up (job=${PBS_JOBID:-interactive}, mode=$MODE, run_dir=$RUN_DIR)"

  # Never start work while the previous chain job (and its Ollama/experiment
  # tree) may still be alive — especially if we land on the same node.
  local pred="${SVO69_PREDECESSOR:-}"
  [[ -z "$pred" || "$pred" == "none" ]] && pred="$(chain_get predecessor)"
  if [[ -n "${pred:-}" && "$pred" != "none" ]]; then
    wait_predecessor "$pred"
  fi
  chain_set "predecessor" "none"

  if is_done; then
    log "DONE mode=$MODE already in run.log — nothing to do; exiting (chain ends)"
    exit 0
  fi

  # Torch EARLY: the chain must survive even if this allocation dies at the
  # very next instant. submit_successor is guarded against duplicates.
  submit_successor || log "WARN: successor submission failed — will retry in the watch loop"

  ensure_ollama || { log "FATAL: Ollama unavailable — exiting (successor will retry)"; exit 1; }

  if run_alive; then
    log "run script already alive — attaching"
  else
    start_run || log "WARN: initial launch failed; watch loop will retry"
  fi

  local session=0 attempts rem
  while true; do
    if is_done; then
      log "DONE mode=$MODE reached — exiting cleanly (queued successor will no-op)"
      exit 0
    fi
    if ! run_alive; then
      if (( session < SESSION_MAX )); then
        session=$((session + 1))
        local prev
        prev=$(chain_get attempts); prev="${prev:-0}"
        attempts=$(( prev + 1 ))
        chain_set "attempts" "$attempts"
        if (( attempts > GLOBAL_MAX )); then
          log "FATAL-STOP: global restart cap ($GLOBAL_MAX) exceeded (attempt #$attempts) — halting chain for human inspection"
          qdel_chain_jobs
          exit 1
        fi
        diagnose_and_restart
        continue
      else
        log "session restart cap ($SESSION_MAX) reached — idling until walltime (successor will take over)"
      fi
    fi
    # Keep the torch alive (e.g. if the early submission failed).
    if [[ -z "$(chain_job_ids)" ]]; then
      submit_successor || log "WARN: successor submission failed (will retry)"
    fi
    rem=$(walltime_remaining_s)
    case "$rem" in (''|*[!0-9]*) rem=-1 ;; esac   # harden against empty/garbage
    if (( rem >= 0 && rem < RESERVE )); then
      log "walltime remaining ${rem}s < reserve ${RESERVE}s — exiting; PBS kill follows, successor resumes"
      exit 0
    fi
    sleep "$POLL"
  done
}

cmd_status() {
  echo "== SWO69 continuation status =="
  echo "run dir : $RUN_DIR"
  echo "mode    : $MODE"
  if is_done; then echo "state   : DONE"
  elif run_alive; then echo "state   : RUNNING"
  else echo "state   : NOT RUNNING"; fi
  local ls_ le_
  ls_=$(grep -F ' START ' "$RUN_LOG" 2>/dev/null | tail -1); [[ -n "$ls_" ]] && echo "last start: $ls_"
  le_=$(grep -F ' END   ' "$RUN_LOG" 2>/dev/null | tail -1); [[ -n "$le_" ]] && echo "last end  : $le_"
  echo "done conds: $(find "$RUN_DIR" -path '*/per_question/*.jsonl' -size +0c 2>/dev/null | wc -l)"
  if curl -sf --max-time 3 "$OLLAMA_URL/api/version" >/dev/null 2>&1; then
    echo "ollama  : up"
  else
    echo "ollama  : DOWN"
  fi
  local rem
  rem=$(walltime_remaining_s)
  [[ "$rem" == "-1" ]] && echo "walltime: not inside a PBS job" || echo "walltime remaining: ${rem}s"
  local ids
  ids=$(chain_job_ids)
  if [[ -n "$ids" ]]; then echo "chain jobs:"; sed 's/^/  /' <<<"$ids"; else echo "chain jobs: none in queue"; fi
  echo "restart attempts (total): $(chain_get attempts)"
}

cmd_submit() {
  lock_acquire || exit 1
  submit_successor
}

case "${1:-run}" in
  run)    cmd_run ;;
  status) cmd_status ;;
  submit) cmd_submit ;;
  *) echo "usage: $0 run|status|submit" >&2; exit 2 ;;
esac
