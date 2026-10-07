#!/usr/bin/env bash
# Reproduce the tested local research prerequisites without changing tracked files.
set -euo pipefail
cd /workspace/apertus
export UV_CACHE_DIR=/workspace/.cache/uv
if [[ ! -x .venv/bin/python ]]; then
    uv venv .venv
fi
uv pip install --python .venv/bin/python --require-hashes -r track_2a/requirements-pdf.lock
make test PYTHON=/workspace/apertus/.venv/bin/python
cd track_2a
export PYTHONPATH=src
if [[ ! -f data/private/ost-reference.jsonl ]]; then
    ../.venv/bin/python -m ost_nli prepare-ost --output data/private/ost-reference.jsonl
fi
if [[ ! -f data/private/splits-strict/manifest.json ]]; then
    ../.venv/bin/python -m ost_nli split data/private/ost-reference.jsonl --output-dir data/private/splits-strict --seed 42
fi
../.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
from ost_nli.data import fingerprint,load_dataset,words
from ost_nli.official import SOURCE_SHA256
source=Path('data/private/official/v1.1.jsonl')
if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=SOURCE_SHA256:
    raise SystemExit('Pinned official source is missing or changed; restore verified source before experiments')
saved=json.loads(Path('experiments/split_manifest.json').read_text())
actual=json.loads(Path('data/private/splits-strict/manifest.json').read_text())
if saved!=actual:
    raise SystemExit('Frozen split manifest mismatch; do not overwrite user data or silently resplit')
if fingerprint(load_dataset('data/private/ost-reference.jsonl'))!=saved['dataset_sha256']:
    raise SystemExit('Canonical dataset fingerprint mismatch')
groups,claims=[],[]
for name,part in saved['partitions'].items():
    rows=load_dataset(f'data/private/splits-strict/{name}.jsonl')
    if fingerprint(rows)!=part['sha256']:
        raise SystemExit(f'{name} partition fingerprint mismatch')
    groups.append({r['booklet_id'] for r in rows})
    claims.append({' '.join(words(r['claim'])) for r in rows})
for i in range(3):
    for j in range(i):
        if groups[i]&groups[j] or claims[i]&claims[j]:
            raise SystemExit('Booklet or duplicate-claim leakage detected')
print('Official checksum, canonical data and strict split fingerprints verified')
PY
../.venv/bin/python -m ost_nli retrieve data/example-booklet.json 'jährlicher Beitrag 100 Franken' --k 1
bash scripts/cloud-docker.sh build
