#!/usr/bin/env bash
# Offline runner: invoke a main class from the prebuilt shadow jar directly,
# bypassing ./gradlew (this environment has no external network, so the
# Gradle wrapper cannot download its distribution).
#
# Usage: scripts/run_kt.sh <main-class> <args...>
#   e.g. scripts/run_kt.sh shared.eval.graph.MultiConditionGraphExperimentKt \
#         --data data/... --output-dir eval_results/...
#
# The jar must be up to date with the sources; if you edit Kotlin, rebuild it
# on a machine with network (./gradlew shadowJar) before using this wrapper.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAR="$ROOT/build/libs/causalrag-0.0.1-all.jar"

MAIN_CLASS="${1:?usage: run_kt.sh <main-class> <args...>}"
shift

if [[ ! -f "$JAR" ]]; then
  echo "error: $JAR not found; build it first (./gradlew shadowJar)" >&2
  exit 1
fi

cd "$ROOT"
exec java -cp "$JAR" "$MAIN_CLASS" "$@"
