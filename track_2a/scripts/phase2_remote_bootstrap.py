"""Bounded phase2 worker: native controls, CPU head, export; no holdout or LoRA."""
import json
import os
import subprocess
from pathlib import Path


def run(command):
    subprocess.run(command,check=True)


def main():
    token = os.environ.pop('HF_TOKEN','')
    if not token:
        raise ValueError('Provider-managed HF_TOKEN was not injected')
    os.environ.update(PYTHONPATH='track_2a/src',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
                      HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HOME='/workspace/hf',
                      OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for lock in ('requirements-v15-cuda.lock','requirements-v15.lock','requirements-head-v15.lock','requirements-pdf.lock'):
        run(['python','-m','pip','install','--no-cache-dir','--require-hashes','-r','track_2a/'+lock])
    run(['python','track_2a/scripts/check_v15_cuda.py'])
    from ost_nli.v15 import download
    try:
        download('/workspace/model',token)
    finally:
        del token
    print('PHASE2_MODEL_ORIGINAL_SHARDS_VERIFIED',flush=True)
    source = Path('/workspace/phase2-source')
    full = Path('/workspace/phase2-booklets')
    inputs = Path('/workspace/phase2-inputs')
    root = Path('track_2a/experiments/apertus-v15-phase2-controls-v1')
    run(['python','track_2a/scripts/prepare_phase2_source.py','--output',str(source)])
    run(['python','track_2a/scripts/prepare_booklets.py','--source',str(source/'selected-official.jsonl'),
         '--splits',str(source),'--output',str(full),'--partitions','train','validation'])
    from ost_nli.data import load_dataset, fingerprint
    expected = {'train':'73dfb46269072ba2c2451c7810faf91b6fccfe658a09afb0d82ff83b4bf45cd5',
                'validation':'00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f'}
    for split,digest in expected.items():
        if fingerprint(load_dataset(full/f'{split}.jsonl')) != digest:
            raise ValueError('Original full train/validation fingerprint changed')
    run(['python','track_2a/scripts/prepare_phase2_inputs.py','--data',str(full),'--output',str(inputs)])
    run(['python','track_2a/scripts/run_phase2_controls.py','--model-dir','/workspace/model',
         '--inputs',str(inputs),'--output',str(root)])
    run(['python','track_2a/scripts/train_v15_head.py','--cache',str(root/'claim-only'),
         '--context','claim-only','--features','hidden','--train',str(full/'train.jsonl'),
         '--validation',str(full/'validation.jsonl'),'--output-dir',str(root/'claim-only-fitted-head')])
    for name,path in [('inputs_manifest.json',inputs/'manifest.json'),('source_audit.json',source/'audit.json'),
                      ('booklets_audit.json',full/'audit.json')]:
        (root/name).write_bytes(path.read_bytes())
    run(['python','track_2a/scripts/export_vast_cache.py','--relative-root',str(root)])
    print('PHASE2_CONTROL_RESULTS_EXPORTED',flush=True)


if __name__ == '__main__':
    main()
