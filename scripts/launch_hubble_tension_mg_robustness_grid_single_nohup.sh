#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/launch_hubble_tension_mg_robustness_grid_single_nohup.sh <phase>
  scripts/launch_hubble_tension_mg_robustness_grid_single_nohup.sh <out_root> <phase>

Phases:
  smoke | pilot | full

Examples:
  scripts/launch_hubble_tension_mg_robustness_grid_single_nohup.sh smoke
  scripts/launch_hubble_tension_mg_robustness_grid_single_nohup.sh outputs/hubble_tension_mg_robustness_full full
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
    out_root="outputs/hubble_tension_mg_robustness_${phase}_${timestamp}"
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

n_total="$(nproc)"
if [ "$n_total" -lt 1 ]; then
  n_total=1
fi
cpuset="${CPUSET:-0-$((n_total - 1))}"

run_dirs="${RUN_DIRS:-outputs/finalization/highpower_multistart_v2/M0_start101,outputs/finalization/highpower_multistart_v2/M0_start202,outputs/finalization/highpower_multistart_v2/M0_start303,outputs/finalization/highpower_multistart_v2/M0_start404,outputs/finalization/highpower_multistart_v2/M0_start505}"
sigma_highz_fracs="${SIGMA_HIGHZ_FRACS:-0.005,0.01,0.02}"
h0_local_refs="${H0_LOCAL_REFS:-72,73,74}"
local_modes="${LOCAL_MODES:-external,truth}"
gr_omega_modes="${GR_OMEGA_MODES:-sample,fixed}"
gr_omega_fixed="${GR_OMEGA_FIXED:-0.315}"

draws=4096
n_rep=5000
z_max=0.62
z_n=240
z_anchors="0.2,0.35,0.5,0.62"
h0_local_sigma="${H0_LOCAL_SIGMA:-1.0}"
h0_planck_ref="${H0_PLANCK_REF:-67.4}"
h0_planck_sigma="${H0_PLANCK_SIGMA:-0.5}"
omega_m_planck="${OMEGA_M_PLANCK:-0.315}"
heartbeat_sec=60

case "$phase" in
  smoke)
    run_dirs="outputs/finalization/highpower_multistart_v2/M0_start101"
    sigma_highz_fracs="0.01"
    h0_local_refs="73"
    local_modes="external,truth"
    gr_omega_modes="sample,fixed"
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
  "cpuset": "$cpuset",
  "n_total_cores": $n_total,
  "run_dirs": "$run_dirs",
  "sigma_highz_fracs": "$sigma_highz_fracs",
  "h0_local_refs": "$h0_local_refs",
  "local_modes": "$local_modes",
  "gr_omega_modes": "$gr_omega_modes",
  "gr_omega_fixed": $gr_omega_fixed,
  "draws": $draws,
  "n_rep": $n_rep,
  "z_max": $z_max,
  "z_n": $z_n,
  "z_anchors": "$z_anchors",
  "h0_local_sigma": $h0_local_sigma,
  "h0_planck_ref": $h0_planck_ref,
  "h0_planck_sigma": $h0_planck_sigma,
  "omega_m_planck": $omega_m_planck,
  "heartbeat_sec": $heartbeat_sec
}
JSON

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"
echo "[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[job] phase=$phase out_root=$out_root"
PYTHONPATH=src .venv/bin/python scripts/run_hubble_tension_mg_forecast_robustness_grid.py \
  --out-root "$out_root" \
  --run-dirs "$run_dirs" \
  --sigma-highz-fracs "$sigma_highz_fracs" \
  --h0-local-refs "$h0_local_refs" \
  --local-modes "$local_modes" \
  --gr-omega-modes "$gr_omega_modes" \
  --gr-omega-fixed "$gr_omega_fixed" \
  --draws "$draws" \
  --n-rep "$n_rep" \
  --seed0 1000 \
  --z-max "$z_max" \
  --z-n "$z_n" \
  --z-anchors "$z_anchors" \
  --h0-local-sigma "$h0_local_sigma" \
  --h0-planck-ref "$h0_planck_ref" \
  --h0-planck-sigma "$h0_planck_sigma" \
  --omega-m-planck "$omega_m_planck" \
  --heartbeat-sec "$heartbeat_sec"
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
echo "[launcher] status: tail -n 100 \"$run_log\""
