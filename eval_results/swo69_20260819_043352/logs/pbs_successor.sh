#!/bin/sh
#PBS -N swo69_batch
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -P gah51681
# Unique output per predecessor (this PBS does not expand %j)
#PBS -o /home/aad13623fe/CausalHippo/eval_results/swo69_20260819_043352/logs/pbs_from_2170104.pbs1.out
PATH=/opt/pbs/bin:$PATH
module load cuda/12.8
export JAVA_HOME=/home/aad13623fe/jdk-21.0.6
export PATH=$JAVA_HOME/bin:$PATH
export GRADLE_USER_HOME=/tmp/ugai/gradle
export SVO69_TS=20260819_043352
export SVO69_MODE=all
export SVO69_PREDECESSOR=2170104.pbs1
cd /home/aad13623fe/CausalHippo
exec bash SWO69/continue_swo69.sh run
