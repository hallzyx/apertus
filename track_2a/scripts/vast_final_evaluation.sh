#!/usr/bin/env bash
set -euo pipefail
cd /workspace/apertus
export PYTHONPATH=track_2a/src HF_HOME=/workspace/hf HF_HUB_DISABLE_XET=1
export_results() {
    task_status=$?
    trap - EXIT TERM
    python track_2a/scripts/export_vast_cache.py --relative-root track_2a/experiments/apertus-final-test-v1 || true
    echo "APERTUS_FINAL_WORKER_EXIT=$task_status"
    exit "$task_status"
}
trap export_results EXIT
trap 'exit 143' TERM
python -m pip install --no-input transformers==4.56.2 huggingface-hub==0.35.3 safetensors==0.6.2 numpy==2.2.6
python -m ost_nli prepare-ost --source track_2a/data/private/official/v1.1.jsonl --output track_2a/data/private/ost-reference.jsonl
python -m ost_nli split track_2a/data/private/ost-reference.jsonl --output-dir track_2a/data/private/splits-strict --seed 42
python track_2a/scripts/download_cpu_model.py --model-dir /workspace/model
python track_2a/scripts/final_frozen_evaluation.py --model-dir /workspace/model --head-dir track_2a/deployment --splits track_2a/data/private/splits-strict --output-dir track_2a/experiments/apertus-final-test-v1 --device cuda
