#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export PIPELINE_MODE=${PIPELINE_MODE:-full-train}
export RUN_TAG=${RUN_TAG:-cura_astar_genarm_v1}
export CONFIG=${CONFIG:-Method/CURA/configs/sentiment_astar_genarm.json}
export TARGET_EVALUATOR=${TARGET_EVALUATOR:-models/distilbert-sst2}
export SIGNAL_BUDGET=${SIGNAL_BUDGET:-3}
export OUTPUT=${OUTPUT:-results/cura_astar_genarm_v1.json}

exec "$SCRIPT_DIR/run_cura_astar_pipeline.sh"
