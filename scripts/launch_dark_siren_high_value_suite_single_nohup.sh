#!/usr/bin/env bash
set -euo pipefail

# Robust detached launcher for high-value dark-siren suite:
#   (1) selection-marginalized hierarchy
#   (2) blind GR-vs-MU challenge
#   (3) independent cross-cache replication
#
# Usage:
#   scripts/launch_dark_siren_high_value_suite_single_nohup.sh [mode]
#   scripts/launch_dark_siren_high_value_suite_single_nohup.sh [out_root] [mode]
#
# Modes: smoke | pilot | full

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

arg1="${1:-full}"
arg2="${2:-}"
if [[ -n "$arg2" ]]; then
  out_root="$arg1"
  mode="$arg2"
else
  mode="$arg1"
  stamp="$(date -u +%Y%m%d_%H%M%SUTC)"
  out_root="outputs/dark_siren_high_value_suite_${mode}_${stamp}"
fi

case "$mode" in
  smoke|pilot|full) ;;
  *)
    echo "Unknown mode '$mode' (expected smoke|pilot|full)"
    exit 2
    ;;
esac

mkdir -p "$out_root"

CPUSET="${CPUSET:-0-$(($(nproc)-1))}"
NPROC="${NPROC:-$(nproc)}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
SEED="${SEED:-0}"
RUN_DIR="${RUN_DIR:-outputs/finalization/highpower_multistart_v2/M0_start101}"
MU_LOGR_SCALE="${MU_LOGR_SCALE:-1.0}"

cat > "$out_root/job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
.venv/bin/python scripts/run_dark_siren_high_value_suite.py \\
  --out-root "$out_root" \\
  --mode "$mode" \\
  --run-dir "$RUN_DIR" \\
  --seed "$SEED" \\
  --n-proc "$NPROC" \\
  --heartbeat-sec "$HEARTBEAT_SEC" \\
  --mu-logr-scale "$MU_LOGR_SCALE"
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$out_root/job.sh"

cat > "$out_root/launcher_manifest.json" <<EOF
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "mode": "$mode",
  "out_root": "$out_root",
  "cpuset": "$CPUSET",
  "nproc": $NPROC,
  "heartbeat_sec": $HEARTBEAT_SEC,
  "seed": $SEED,
  "run_dir": "$RUN_DIR",
  "mu_logr_scale": $MU_LOGR_SCALE
}
EOF

env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  setsid taskset -c "$CPUSET" bash "$out_root/job.sh" > "$out_root/run.log" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$out_root/pid.txt"
echo "[launch] mode=$mode pid=$pid out_root=$out_root cpuset=$CPUSET nproc=$NPROC"

