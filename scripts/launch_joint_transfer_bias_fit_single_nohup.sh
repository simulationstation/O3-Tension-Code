#!/usr/bin/env bash
set -euo pipefail

# Robust detached launcher for joint SN+BAO+CC+O3 transfer-bias fits.
#
# Usage:
#   scripts/launch_joint_transfer_bias_fit_single_nohup.sh [mode]
#   scripts/launch_joint_transfer_bias_fit_single_nohup.sh [out_root] [mode]
#
# Modes:
#   smoke   - fast sanity run
#   pilot   - decision-grade intermediate run
#   full    - larger production run
#   recover - synthetic parameter-recovery sanity run

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

arg1="${1:-smoke}"
arg2="${2:-}"

if [[ -n "$arg2" ]]; then
  out_root="$arg1"
  mode="$arg2"
else
  mode="$arg1"
  timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
  out_root="outputs/joint_transfer_bias_fit_${mode}_${timestamp}"
fi

case "$mode" in
  smoke|pilot|full|recover) ;;
  *)
    echo "Unknown mode: $mode"
    exit 2
    ;;
esac

mkdir -p "$out_root"

CPUSET="${CPUSET:-0-$(($(nproc)-1))}"
WORKERS="${WORKERS:-$(nproc)}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-60}"
O3_DELTA_LPD="${O3_DELTA_LPD:-3.669945265}"

DRAWs_TOTAL=512
THETA_SAMPLES=512
THETA_CHUNK=64
MAX_CHUNKS=0
SEED="${SEED:-0}"
SYNTH_ARG=()

if [[ "$mode" == "pilot" ]]; then
  DRAWs_TOTAL=2048
  THETA_SAMPLES=4096
  THETA_CHUNK=256
elif [[ "$mode" == "full" ]]; then
  DRAWs_TOTAL=8192
  THETA_SAMPLES=32768
  THETA_CHUNK=512
elif [[ "$mode" == "recover" ]]; then
  DRAWs_TOTAL=1024
  THETA_SAMPLES=2048
  THETA_CHUNK=128
  synth_json="$out_root/synthetic_truth.json"
  cat > "$synth_json" <<'JSON'
{
  "draw_index": 0,
  "beta_ia": 0.02,
  "beta_cc": -0.05,
  "beta_bao": 0.01,
  "delta_h0_ladder": 0.6,
  "seed": 12345
}
JSON
  SYNTH_ARG=(--synthetic-theta-json "$synth_json")
fi

chunks=$(( (THETA_SAMPLES + THETA_CHUNK - 1) / THETA_CHUNK ))
if [[ "$MAX_CHUNKS" -gt 0 && "$MAX_CHUNKS" -lt "$chunks" ]]; then
  chunks="$MAX_CHUNKS"
fi
if [[ "$WORKERS" -gt "$chunks" ]]; then
  WORKERS="$chunks"
fi

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
.venv/bin/python scripts/run_joint_transfer_bias_fit.py \\
  --out "$out_root" \\
  --draws-total "$DRAWs_TOTAL" \\
  --theta-samples "$THETA_SAMPLES" \\
  --theta-chunk "$THETA_CHUNK" \\
  --workers "$WORKERS" \\
  --seed "$SEED" \\
  --o3-delta-lpd "$O3_DELTA_LPD" \\
  --heartbeat-sec "$HEARTBEAT_SEC" \\
  --resume \\
  \${MAX_CHUNKS:+--max-chunks "$MAX_CHUNKS"} \\
  ${SYNTH_ARG[*]}
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$out_root/job.sh"

cat > "$out_root/launcher_manifest.json" <<EOF
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "mode": "$mode",
  "out_root": "$out_root",
  "cpuset": "$CPUSET",
  "workers": $WORKERS,
  "draws_total": $DRAWs_TOTAL,
  "theta_samples": $THETA_SAMPLES,
  "theta_chunk": $THETA_CHUNK,
  "seed": $SEED,
  "heartbeat_sec": $HEARTBEAT_SEC
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
echo "[launch] mode=$mode pid=$pid out_root=$out_root cpuset=$CPUSET workers=$WORKERS"
