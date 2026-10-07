#!/usr/bin/env bash
# Task-owned lease only. No account or model credentials are stored here.
set -euo pipefail
cd /workspace/apertus
export PYTHONPATH=track_2a/src HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HOME=/workspace/hf
python -m pip install --no-cache-dir --require-hashes -r track_2a/requirements-v15.lock
python - <<'PY'
import torch
from transformers import Apertus1p5ForConditionalGeneration
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print('V15_NATIVE_READY',torch.__version__,torch.cuda.get_device_name(0),flush=True)
PY
for attempt in $(seq 1 360); do
    if [[ -f /workspace/model/verified_manifest.json && -f /workspace/v15-inputs/manifest.json ]]; then break; fi
    sleep 5
done
[[ -f /workspace/model/verified_manifest.json && -f /workspace/v15-inputs/manifest.json ]]
python track_2a/scripts/run_v15_experiments.py --model-dir /workspace/model \
    --inputs /workspace/v15-inputs --output /workspace/v15-validation --mode validation
context=$(python - <<'PY'
import json
from pathlib import Path
values=[]
for context in ['full','bm25','dense','hybrid']:
    metrics=json.loads((Path('/workspace/v15-validation')/(context+'-score')/'metrics.json').read_text())
    values.append((metrics['macro_f1'],-metrics['average_context_tokens'],context))
choice=max(values)[-1]
Path('/workspace/v15-context-choice.json').write_text(json.dumps({'context':choice,'selection':'Highest validation Macro-F1 among booklet-only full/retrieval class-score baselines; token count breaks ties. Test unused.'})+'\n')
print(choice)
PY
)
python track_2a/scripts/run_v15_experiments.py --model-dir /workspace/model \
    --inputs /workspace/v15-inputs --output /workspace/v15-train --mode train --context "$context"
echo V15_VALIDATION_AND_TRAIN_READY
