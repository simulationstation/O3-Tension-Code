#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_hubble_tension_mg_forecast_single_nohup.sh <phase>
  scripts/launch_hubble_tension_mg_forecast_single_nohup.sh <out_root> <phase>

Phases:
  smoke | pilot | full

Examples:
  scripts/launch_hubble_tension_mg_forecast_single_nohup.sh smoke
  scripts/launch_hubble_tension_mg_forecast_single_nohup.sh outputs/hubble_tension_mg_forecast_full full
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
  smoke|pilot|full)
    phase="$1"
    out_root="outputs/hubble_tension_mg_forecast_${phase}_${timestamp}"
    ;;
  *)
    out_root="$1"
    phase="${2:-pilot}"
    ;;
esac

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)." >&2
  exit 2
fi

run_dir="${RUN_DIR:-outputs/finalization/highpower_multistart_v2/M0_start101}"
n_total="$(nproc)"
if [ "$n_total" -lt 1 ]; then
  n_total=1
fi
cpuset="${CPUSET:-0-$((n_total - 1))}"

draws=4096
n_rep=5000
z_max="${Z_MAX:-0.62}"
z_n=240
z_anchors="${Z_ANCHORS:-0.2,0.35,0.5,0.62}"
sigma_highz_frac=0.01
local_mode="${LOCAL_MODE:-external}"
h0_local_ref="${H0_LOCAL_REF:-73.0}"
h0_local_sigma="${H0_LOCAL_SIGMA:-1.0}"
h0_planck_ref="${H0_PLANCK_REF:-67.4}"
h0_planck_sigma="${H0_PLANCK_SIGMA:-0.5}"
omega_m_planck="${OMEGA_M_PLANCK:-0.315}"
gr_omega_mode="${GR_OMEGA_MODE:-sample}"
gr_omega_fixed="${GR_OMEGA_FIXED:-0.315}"
heartbeat_sec=60

case "$phase" in
  smoke)
    draws=1024
    n_rep=500
    z_n=160
    heartbeat_sec=10
    ;;
  pilot)
    draws=4096
    n_rep=5000
    z_n=240
    heartbeat_sec=30
    ;;
  full)
    draws=8192
    n_rep=20000
    z_n=320
    heartbeat_sec=30
    ;;
  *)
    echo "ERROR: unknown phase '$phase' (expected smoke|pilot|full)." >&2
    exit 2
    ;;
esac

mkdir -p "$out_root"
job_sh="$out_root/job.sh"
run_log="$out_root/run.log"
pid_file="$out_root/pid.txt"
manifest="$out_root/launcher_manifest.json"

cat > "$manifest" <<JSON
{
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "repo_root": "$repo_root",
  "out_root": "$out_root",
  "phase": "$phase",
  "run_dir": "$run_dir",
  "cpuset": "$cpuset",
  "n_total_cores": $n_total,
  "draws": $draws,
  "n_rep": $n_rep,
  "z_max": $z_max,
  "z_n": $z_n,
  "z_anchors": "$z_anchors",
  "sigma_highz_frac": $sigma_highz_frac,
  "local_mode": "$local_mode",
  "h0_local_ref": $h0_local_ref,
  "h0_local_sigma": $h0_local_sigma,
  "h0_planck_ref": $h0_planck_ref,
  "h0_planck_sigma": $h0_planck_sigma,
  "omega_m_planck": $omega_m_planck,
  "gr_omega_mode": "$gr_omega_mode",
  "gr_omega_fixed": $gr_omega_fixed,
  "heartbeat_sec": $heartbeat_sec
}
JSON

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[job] phase=$phase out_root=$out_root run_dir=$run_dir"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_mg_forecast.py \
  --run-dir "$run_dir" \
  --out "$out_root" \
  --draws "$draws" \
  --seed 0 \
  --z-max "$z_max" \
  --z-n "$z_n" \
  --z-anchors "$z_anchors" \
  --n-rep "$n_rep" \
  --sigma-highz-frac "$sigma_highz_frac" \
  --local-mode "$local_mode" \
  --h0-local-ref "$h0_local_ref" \
  --h0-local-sigma "$h0_local_sigma" \
  --h0-planck-ref "$h0_planck_ref" \
  --h0-planck-sigma "$h0_planck_sigma" \
  --omega-m-planck "$omega_m_planck" \
  --gr-omega-mode "$gr_omega_mode" \
  --gr-omega-fixed "$gr_omega_fixed" \
  --heartbeat-sec "$heartbeat_sec" \
  --resume
echo "[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$job_sh"

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
echo "[launcher] pid=$pid (written to $pid_file)"
echo "[launcher] run_log=$run_log"
echo "[launcher] status: tail -n 80 \"$run_log\""
