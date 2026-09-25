#!/usr/bin/env bash
# run_cura_3signal_genarm.sh
#
# Full-train pipeline cho CURA 3-signal: RAD + CD-Q + sentiment-GenARM
# Dùng config: Method/CURA/configs/sentiment_astar_genarm.json
#
# Cách chạy (từ repo root):
#   bash Method/CURA/scripts/run_cura_3signal_genarm.sh
#
# Override ví dụ:
#   PIPELINE_MODE=inference-tune bash Method/CURA/scripts/run_cura_3signal_genarm.sh
#   DRY_RUN=1 bash Method/CURA/scripts/run_cura_3signal_genarm.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export PIPELINE_MODE=${PIPELINE_MODE:-full-train}
export RUN_TAG=${RUN_TAG:-cura_3signal_genarm_v1}
export CONFIG=${CONFIG:-Method/CURA/configs/sentiment_astar_genarm.json}
export TARGET_EVALUATOR=${TARGET_EVALUATOR:-models/distilbert-sst2}
export SIGNAL_BUDGET=${SIGNAL_BUDGET:-3}
export OUTPUT=${OUTPUT:-results/cura_3signal_genarm_v1.json}

exec "$SCRIPT_DIR/run_cura_astar_pipeline.sh"
