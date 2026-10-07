#!/usr/bin/env bash
set -euo pipefail
cd /workspace/apertus
export_results() {
    status=$?
    trap - EXIT TERM
    python track_2a/scripts/export_vast_cache.py
    echo "APERTUS_CACHE_WORKER_EXIT=$status"
    exit "$status"
}
trap export_results EXIT
trap 'exit 143' TERM
export PYTHONPATH=track_2a/src HF_HOME=/workspace/hf
python -m pip install --no-cache-dir 'transformers==4.56.2' 'huggingface-hub==0.35.3' 'safetensors==0.6.2' 'numpy==2.2.6'
python -m ost_nli prepare-ost --output track_2a/data/private/ost-reference.jsonl
python -m ost_nli split track_2a/data/private/ost-reference.jsonl --output-dir track_2a/data/private/splits-strict --seed 42
python - <<'PY'
import json
from pathlib import Path
if json.loads(Path('track_2a/experiments/split_manifest.json').read_text()) != json.loads(Path('track_2a/data/private/splits-strict/manifest.json').read_text()):
    raise SystemExit('Frozen split manifest mismatch')
PY
python track_2a/scripts/download_cpu_model.py --model-dir /workspace/model
python track_2a/scripts/cache_apertus_features.py --model-dir /workspace/model \
    --train track_2a/data/private/splits-strict/train.jsonl \
    --validation track_2a/data/private/splits-strict/validation.jsonl \
    --output-root track_2a/experiments/apertus-frozen-cache-v1
