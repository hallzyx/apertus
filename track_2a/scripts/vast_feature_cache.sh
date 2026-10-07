#!/usr/bin/env bash
set -euo pipefail
cd /workspace/apertus
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
python - <<'PY'
import base64,hashlib,json,tarfile
from pathlib import Path
archive=Path('/workspace/feature-cache.tar.gz')
with tarfile.open(archive,'w:gz') as bundle:
    bundle.add('track_2a/experiments/apertus-frozen-cache-v1',arcname='track_2a/experiments/apertus-frozen-cache-v1')
data=archive.read_bytes()
encoded=base64.b64encode(data)
root=Path('/workspace/cache-export');root.mkdir()
parts=[]
for index,offset in enumerate(range(0,len(encoded),400_000)):
    part=encoded[offset:offset+400_000]
    name=f'part-{index:04d}.b64'
    content=b'\n'.join(part[i:i+400] for i in range(0,len(part),400))+b'\n'
    (root/name).write_bytes(content)
    parts.append({'name':name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
manifest={'archive_sha256':hashlib.sha256(data).hexdigest(),'archive_bytes':len(data),'parts':parts}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('APERTUS_CACHE_READY='+json.dumps({'archive_bytes':len(data),'parts':len(parts),'sha256':manifest['archive_sha256']}),flush=True)
PY
