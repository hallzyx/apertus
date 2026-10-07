"""Bounded context efficiency worker after verified grounding gate; no holdout."""
import json
import os
from pathlib import Path
from phase2_remote_bootstrap import run


def main():
    gate=json.loads(Path('track_2a/experiments/apertus-v15-phase2/grounding-results.json').read_text())
    if gate['grounding_gate']!='GO_WITH_LIMITATIONS':raise ValueError('Grounding gate did not pass')
    token=os.environ.pop('HF_TOKEN','')
    if not token:raise ValueError('Provider-managed HF_TOKEN required')
    os.environ.update(PYTHONPATH='track_2a/src',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
         HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HOME='/workspace/hf',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for lock in ('requirements-v15-cuda.lock','requirements-v15.lock','requirements-head-v15.lock','requirements-pdf.lock'):
        run(['python','-m','pip','install','--no-cache-dir','--require-hashes','-r','track_2a/'+lock])
    run(['python','track_2a/scripts/check_v15_cuda.py'])
    from ost_nli.v15 import download
    try:download('/workspace/model',token)
    finally:del token
    from ost_nli.dense import download as download_e5
    download_e5('/workspace/e5')
    print('PHASE2_CONTEXT_MODELS_VERIFIED',flush=True)
    source=Path('/workspace/phase2-source');full=Path('/workspace/phase2-booklets')
    inputs=Path('/workspace/phase2-context-inputs');root=Path('track_2a/experiments/apertus-v15-phase2-context-v1')
    run(['python','track_2a/scripts/prepare_phase2_source.py','--output',str(source)])
    run(['python','track_2a/scripts/prepare_booklets.py','--source',str(source/'selected-official.jsonl'),
         '--splits',str(source),'--output',str(full),'--partitions','train','validation'])
    from ost_nli.data import load_dataset,fingerprint
    expected={'train':'73dfb46269072ba2c2451c7810faf91b6fccfe658a09afb0d82ff83b4bf45cd5',
              'validation':'00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f'}
    for split,digest in expected.items():
        if fingerprint(load_dataset(full/f'{split}.jsonl'))!=digest:raise ValueError('Frozen split changed')
    run(['python','track_2a/scripts/prepare_phase2_context_inputs.py','--data',str(full),
        '--reference-data',str(source),'--model-dir','/workspace/model','--retriever-dir','/workspace/e5','--output',str(inputs)])
    conditions=['hybrid-1k','hybrid-2k','hybrid-4k','hybrid-8k','prefix-2k']
    root.mkdir(parents=True,exist_ok=True)
    (root/'inputs_manifest.json').write_bytes((inputs/'manifest.json').read_bytes())
    try:
        run(['python','track_2a/scripts/run_phase2_controls.py','--model-dir','/workspace/model',
             '--inputs',str(inputs),'--output',str(root),'--study','context','--max-runtime-seconds','7200','--conditions',*conditions])
    except BaseException:
        # Preserve completed chunks before the outer lease trap stops the instance.
        if (root/'experiment.json').exists():
            info=json.loads((root/'experiment.json').read_text());info['status']='interrupted'
            (root/'experiment.json').write_text(json.dumps(info,indent=2)+'\n')
            run(['python','track_2a/scripts/export_vast_cache.py','--relative-root',str(root)])
        raise
    for condition in conditions:
        run(['python','track_2a/scripts/train_v15_head.py','--cache',str(root/condition),
             '--context',condition,'--features','hidden','--train',str(full/'train.jsonl'),
             '--validation',str(full/'validation.jsonl'),'--output-dir',str(root/(condition+'-fitted-head'))])
        # Keep exact evidence/provenance and packing diagnostics, without any gold text.
        values=[json.loads(s) for s in (inputs/f'validation-{condition}.jsonl').read_text().splitlines()]
        for row in values:row.pop('messages');row.pop('claim')
        (root/condition/'validation-evidence.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values))
    for name,path in [('inputs_manifest.json',inputs/'manifest.json'),('source_audit.json',source/'audit.json'),
                      ('booklets_audit.json',full/'audit.json'),('model_manifest.json',Path('/workspace/model/verified_manifest.json')),
                      ('retriever_manifest.json',Path('/workspace/e5/verified_manifest.json'))]:
        (root/name).write_bytes(path.read_bytes())
    run(['python','track_2a/scripts/export_vast_cache.py','--relative-root',str(root),'--exclude-chunks'])
    print('PHASE2_CONTEXT_RESULTS_EXPORTED',flush=True)


if __name__=='__main__':main()
