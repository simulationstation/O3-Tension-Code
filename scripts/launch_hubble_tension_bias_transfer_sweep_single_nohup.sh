#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_hubble_tension_bias_transfer_sweep_single_nohup.sh <phase>
  scripts/launch_hubble_tension_bias_transfer_sweep_single_nohup.sh <out_root> <phase>

Phases:
  smoke | pilot

Examples:
  scripts/launch_hubble_tension_bias_transfer_sweep_single_nohup.sh smoke
  scripts/launch_hubble_tension_bias_transfer_sweep_single_nohup.sh outputs/hubble_tension_bias_transfer_pilot_YYYYMMDD_UTC pilot
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ "$#" -eq 0 ]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

timestamp="$(date -u +%Y%m%d_%H%M%SUTC)"
phase="pilot"
out_root=""
case "${1:-}" in
  smoke|pilot)
    phase="$1"
    out_root="outputs/hubble_tension_bias_transfer_${phase}_${timestamp}"
    ;;
  *)
    out_root="$1"
    phase="${2:-pilot}"
    ;;
esac

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python" >&2
  exit 2
fi

n_total="$(nproc)"
if [ "$n_total" -lt 1 ]; then
  n_total=1
fi
cpuset="${CPUSET:-0-$((n_total - 1))}"

run_dirs="${RUN_DIRS:-outputs/finalization/highpower_multistart_v2/M0_start101,outputs/finalization/highpower_multistart_v2/M0_start202,outputs/finalization/highpower_multistart_v2/M0_start303,outputs/finalization/highpower_multistart_v2/M0_start404,outputs/finalization/highpower_multistart_v2/M0_start505}"
highz_bias_fracs="${HIGHZ_BIAS_FRACS:--0.01,-0.005,0.0,0.005,0.01}"
local_biases="${LOCAL_BIASES:--0.5,0.0,0.5}"
draws=8192
n_rep=40000
seed0=5000
heartbeat_sec=30

case "$phase" in
  smoke)
    run_dirs="outputs/finalization/highpower_multistart_v2/M0_start101"
    highz_bias_fracs="-0.005,0.0,0.005"
    local_biases="-0.25,0.0,0.25"
    draws=4096
    n_rep=12000
    seed0=7000
    heartbeat_sec=10
    ;;
  pilot)
    ;;
  *)
    echo "ERROR: unknown phase '$phase'" >&2
    exit 2
    ;;
esac

mkdir -p "$out_root"
job_sh="$out_root/job.sh"
run_log="$out_root/run.log"
pid_file="$out_root/pid.txt"

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_bias_transfer_sweep.py \\
  --out-root "$out_root" \\
  --run-dirs "$run_dirs" \\
  --highz-bias-fracs="$highz_bias_fracs" \\
  --local-biases="$local_biases" \\
  --draws "$draws" \\
  --n-rep "$n_rep" \\
  --seed0 "$seed0" \\
  --z-max 0.62 \\
  --z-n 320 \\
  --z-anchors "0.2,0.35,0.5,0.62" \\
  --sigma-highz-frac 0.01 \\
  --local-mode external \\
  --h0-local-ref 73.0 \\
  --h0-local-sigma 1.0 \\
  --h0-planck-ref 67.4 \\
  --h0-planck-sigma 0.5 \\
  --omega-m-planck 0.315 \\
  --gr-omega-mode sample \\
  --gr-omega-fixed 0.315 \\
  --heartbeat-sec "$heartbeat_sec"
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$job_sh"

cat > "$out_root/launcher_manifest.json" <<JSON
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "phase": "$phase",
  "out_root": "$out_root",
  "cpuset": "$cpuset",
  "n_total_cores": $n_total,
  "run_dirs": "$run_dirs",
  "highz_bias_fracs": "$highz_bias_fracs",
  "local_biases": "$local_biases",
  "draws": $draws,
  "n_rep": $n_rep,
  "seed0": $seed0
}
JSON

env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 \
  setsid taskset -c "$cpuset" bash "$job_sh" > "$run_log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$pid_file"
echo "[launcher] started phase=$phase"
echo "[launcher] out_root=$out_root"
echo "[launcher] pid=$pid"
echo "[launcher] run_log=$run_log"
