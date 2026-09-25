#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/jupyter-iec2024se10/.conda/envs/genarm/bin/python}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE_B="${BATCH_SIZE_B:-16}"
BOOTSTRAP="${BOOTSTRAP:-2000}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/results/parm_preference_gap/full}"
PIPELINE="${PROJECT_ROOT}/Method/PARM/preference_gap_pilot.py"
LOG_DIR="${RUN_DIR}/logs"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${PIPELINE}" ]]; then
  echo "Pipeline not found: ${PIPELINE}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}"

echo "PARM preference-realization gap — full pilot"
echo "Run directory : ${RUN_DIR}"
echo "Python        : ${PYTHON_BIN}"
echo "Device        : ${DEVICE}"
echo "Evaluator-B BS: ${BATCH_SIZE_B}"
echo "Bootstrap     : ${BOOTSTRAP}"
echo "Target        : 100 prompts x 11 alphas = 1,100 responses"

echo "[1/4] Preparing the deterministic seed-42 prompt manifest..."
"${PYTHON_BIN}" "${PIPELINE}" prepare \
  --run-dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_DIR}/01_prepare.log"

echo "[2/4] Generating PARM responses (resumable per alpha)..."
"${PYTHON_BIN}" "${PIPELINE}" generate \
  --run-dir "${RUN_DIR}" \
  2>&1 | tee "${LOG_DIR}/02_generate.log"

echo "[3/4] Scoring with independent evaluator pairs A and B..."
"${PYTHON_BIN}" "${PIPELINE}" score \
  --run-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --batch-size-b "${BATCH_SIZE_B}" \
  2>&1 | tee "${LOG_DIR}/03_score.log"

echo "[4/4] Computing metrics and prompt-level bootstrap confidence intervals..."
"${PYTHON_BIN}" "${PIPELINE}" analyze \
  --run-dir "${RUN_DIR}" \
  --bootstrap "${BOOTSTRAP}" \
  2>&1 | tee "${LOG_DIR}/04_analyze.log"

echo "Pipeline complete."
echo "Report : ${RUN_DIR}/report.md"
echo "Summary: ${RUN_DIR}/summary.json"
echo "Scores : ${RUN_DIR}/scores.csv"

