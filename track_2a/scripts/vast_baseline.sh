#!/usr/bin/env bash
# Run only on the task-created CUDA instance; contains no account credentials.
set -euo pipefail
cd /workspace/apertus
emit_results() {
    result=$?
    trap - EXIT
    echo "APERTUS_WORKER_EXIT=$result"
    paths=()
    for cap in 1024 4096; do
        path="track_2a/experiments/apertus-8b-gpu-reference-${cap}-v1"
        [[ ! -d "$path" ]] || paths+=("$path")
    done
    if [[ ${#paths[@]} -gt 0 ]]; then
        tar -czf /workspace/results.tar.gz "${paths[@]}"
        printf 'APERTUS_RESULTS_SHA256='
        sha256sum /workspace/results.tar.gz | cut -d ' ' -f 1
        printf 'APERTUS_RESULTS_B64='
        base64 -w 0 /workspace/results.tar.gz
        printf '\n'
    fi
    exit "$result"
}
trap emit_results EXIT
export PYTHONPATH=track_2a/src HF_HOME=/workspace/hf
python -m pip install --no-cache-dir 'transformers==4.56.2' 'huggingface-hub==0.35.3' 'safetensors==0.6.2' 'numpy==2.2.6'
python - <<'PY'
import torch
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit('CUDA/BF16 unavailable; no substitute CPU rental run')
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,flush=True)
PY
python -m ost_nli prepare-ost --output track_2a/data/private/ost-reference.jsonl
python -m ost_nli split track_2a/data/private/ost-reference.jsonl --output-dir track_2a/data/private/splits-strict --seed 42
python - <<'PY'
import json
from pathlib import Path
expected=json.loads(Path('track_2a/experiments/split_manifest.json').read_text())
actual=json.loads(Path('track_2a/data/private/splits-strict/manifest.json').read_text())
if expected != actual:raise SystemExit('Frozen split manifest mismatch')
PY
python track_2a/scripts/download_cpu_model.py --model-dir /workspace/model
for cap in 1024 4096; do
    python track_2a/scripts/cpu_ost_experiment.py \
        --device cuda --model-dir /workspace/model \
        --train track_2a/data/private/splits-strict/train.jsonl \
        --validation track_2a/data/private/splits-strict/validation.jsonl \
        --output-dir "track_2a/experiments/apertus-8b-gpu-reference-${cap}-v1" \
        --train-per-class 10 --reference-token-cap "$cap"
done
tar -czf /workspace/results.tar.gz \
    track_2a/experiments/apertus-8b-gpu-reference-1024-v1 \
    track_2a/experiments/apertus-8b-gpu-reference-4096-v1
sha256sum /workspace/results.tar.gz > /workspace/results.sha256
echo completed > /workspace/DONE
