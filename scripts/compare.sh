#!/usr/bin/env bash
# Train each task at an equal sample budget per seed, then a milestone summary.
# Usage (any directory):  mamba activate sprint && bash scripts/compare.sh
# Override with environment variables, e.g.  SEEDS="1" STEPS_M=400 bash scripts/compare.sh
# Completed runs are skipped on rerun; an interrupted run is moved to $LOG_ROOT/_incomplete and restarted.
set -euo pipefail

TASKS=${TASKS:-"g1_sprint_paper_map g1_sprint"}
SEEDS=${SEEDS:-"1 2 3"}
STEPS_M=${STEPS_M:-400}   # Million environment steps per run; iterations follow from each task's rollout length.
NUM_ENVS=${NUM_ENVS:-4096}
NUM_THREADS=${NUM_THREADS:-10}
RL_DEVICE=${RL_DEVICE:-auto}
LOG_ROOT=${LOG_ROOT:-logs/comparison_v3}
PYTHON=${PYTHON:-python}

cd "$(dirname "$0")/.."
"$PYTHON" -c "import legged_gym" 2>/dev/null || { echo "legged_gym not importable; run 'mamba activate sprint' first" >&2; exit 1; }
[ "${CONDA_DEFAULT_ENV:-}" = "sprint" ] || echo "warning: active environment is '${CONDA_DEFAULT_ENV:-none}', not 'sprint'" >&2
mkdir -p "$LOG_ROOT/_console"

iterations() {  # task -> iterations for STEPS_M million steps
  "$PYTHON" -c "
import math, sys
from legged_gym.envs import task_registry
steps = task_registry.get_cfgs(sys.argv[1])[1].runner.num_steps_per_env
print(math.ceil($STEPS_M * 1e6 / ($NUM_ENVS * steps)))" "$1" 2>/dev/null
}

echo "== prior speed per command"
# shellcheck disable=SC2086
"$PYTHON" -m legged_gym.scripts.diagnose speed --tasks $TASKS | tee "$LOG_ROOT/speed.jsonl"

train() {  # task iterations seed
  local task=$1 iters=$2 seed=$3
  local dir="$LOG_ROOT/$task/seed$seed" out="$LOG_ROOT/_console/${task}_seed$seed.log"
  if [ -f "$dir/model_$((iters - 1)).pt" ]; then
    echo "== skip $task seed $seed (complete)"; return
  fi
  if [ -e "$dir" ]; then
    mkdir -p "$LOG_ROOT/_incomplete"
    mv "$dir" "$LOG_ROOT/_incomplete/${task}_seed${seed}_$(date +%Y%m%d-%H%M%S)"
  fi
  echo "== train $task seed $seed ($iters iterations, ${STEPS_M}M steps) -> $out"
  "$PYTHON" -m legged_gym.scripts.train --task "$task" --headless --seed "$seed" \
    --max_iterations "$iters" --num_envs "$NUM_ENVS" --num_threads "$NUM_THREADS" \
    --rl_device "$RL_DEVICE" --log-dir "$dir" > "$out" 2>&1
}

# Seeds outer: partial results stay paired for an early compare.
for seed in $SEEDS; do
  for task in $TASKS; do
    train "$task" "$(iterations "$task")" "$seed"
  done
done

echo "== compare"
# shellcheck disable=SC2086
"$PYTHON" -m legged_gym.scripts.diagnose compare --logdir "$LOG_ROOT" --tasks $TASKS | tee "$LOG_ROOT/summary.txt"
# shellcheck disable=SC2086
"$PYTHON" -m legged_gym.scripts.diagnose compare --logdir "$LOG_ROOT" --tasks $TASKS --json > "$LOG_ROOT/summary.json"
echo "Summary: $LOG_ROOT/summary.txt (JSON: summary.json)"
