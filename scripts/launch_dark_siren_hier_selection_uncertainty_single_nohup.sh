#!/usr/bin/env bash
set -euo pipefail

# Robust detached launcher for hierarchical dark-siren selection uncertainty integration.
#
# Usage:
#   scripts/launch_dark_siren_hier_selection_uncertainty_single_nohup.sh [mode]
#   scripts/launch_dark_siren_hier_selection_uncertainty_single_nohup.sh [out_root] [mode]
#
# Modes:
#   smoke  - quick validation run
#   pilot  - intermediate run
#   full   - larger decision-grade run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

arg1="${1:-smoke}"
arg2="${2:-}"

if [[ -n "$arg2" ]]; then
  out_root="$arg1"
  mode="$arg2"
else
  mode="$arg1"
  stamp="$(date -u +%Y%m%d_%H%M%SUTC)"
  out_root="outputs/dark_siren_hier_selection_uncertainty_${mode}_${stamp}"
fi

case "$mode" in
  smoke|pilot|full) ;;
  *)
    echo "Unknown mode: $mode" >&2
    exit 2
    ;;
esac

mkdir -p "$out_root"

CPUSET="${CPUSET:-0-$(($(nproc)-1))}"
NPROC="${NPROC:-$(nproc)}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
PARTIAL_WRITE_MIN_SEC="${PARTIAL_WRITE_MIN_SEC:-20}"
RUN_DIR="${RUN_DIR:-outputs/finalization/highpower_multistart_v2/M0_start101}"
VARIANTS_JSON="${VARIANTS_JSON:-}"

N_REP=4
MAX_DRAWS=64
MAX_EVENTS=36
WEIGHT_SAMPLES=128
WEIGHT_KAPPA=200
SKIP_REALDATA=0
SEED="${SEED:-0}"

if [[ "$mode" == "pilot" ]]; then
  N_REP=24
  MAX_DRAWS=256
  WEIGHT_SAMPLES=2000
elif [[ "$mode" == "full" ]]; then
  N_REP=64
  MAX_DRAWS=256
  WEIGHT_SAMPLES=5000
fi

if [[ -n "${N_REP_OVERRIDE:-}" ]]; then N_REP="${N_REP_OVERRIDE}"; fi
if [[ -n "${MAX_DRAWS_OVERRIDE:-}" ]]; then MAX_DRAWS="${MAX_DRAWS_OVERRIDE}"; fi
if [[ -n "${MAX_EVENTS_OVERRIDE:-}" ]]; then MAX_EVENTS="${MAX_EVENTS_OVERRIDE}"; fi
if [[ -n "${WEIGHT_SAMPLES_OVERRIDE:-}" ]]; then WEIGHT_SAMPLES="${WEIGHT_SAMPLES_OVERRIDE}"; fi
if [[ -n "${WEIGHT_KAPPA_OVERRIDE:-}" ]]; then WEIGHT_KAPPA="${WEIGHT_KAPPA_OVERRIDE}"; fi

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
cmd=(.venv/bin/python scripts/run_dark_siren_hier_selection_uncertainty.py
  --out-root "$out_root"
  --run-dir "$RUN_DIR"
  --seed "$SEED"
  --n-rep "$N_REP"
  --n-proc "$NPROC"
  --max-events "$MAX_EVENTS"
  --max-draws "$MAX_DRAWS"
  --heartbeat-sec "$HEARTBEAT_SEC"
  --partial-write-min-sec "$PARTIAL_WRITE_MIN_SEC"
  --weight-kappa "$WEIGHT_KAPPA"
  --weight-samples "$WEIGHT_SAMPLES"
)
if [[ -n "$VARIANTS_JSON" ]]; then
  cmd+=(--variants-json "$VARIANTS_JSON")
fi
if [[ "$SKIP_REALDATA" == "1" ]]; then
  cmd+=(--skip-realdata)
fi
if [[ -n "\${REAL_PE_RECORD_IDS:-}" ]]; then
  cmd+=(--real-pe-record-ids "\${REAL_PE_RECORD_IDS}")
fi
"\${cmd[@]}"
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$out_root/job.sh"

cat > "$out_root/launcher_manifest.json" <<EOF
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "mode": "$mode",
  "out_root": "$out_root",
  "cpuset": "$CPUSET",
  "n_proc": $NPROC,
  "n_rep": $N_REP,
  "max_draws": $MAX_DRAWS,
  "max_events": $MAX_EVENTS,
  "weight_samples": $WEIGHT_SAMPLES,
  "weight_kappa": $WEIGHT_KAPPA,
  "skip_realdata": $SKIP_REALDATA,
  "run_dir": "$RUN_DIR"
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
echo "[launch] mode=$mode pid=$pid out_root=$out_root cpuset=$CPUSET n_proc=$NPROC"
