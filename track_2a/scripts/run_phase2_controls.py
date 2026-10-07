"""Resumable actual native Apertus grounding controls; train/validation only."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    from ost_nli.v15 import Engine, REVISION
    from ost_nli.metrics import evaluate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--inputs', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    p.add_argument('--study', default='grounding', choices=['grounding','context'])
    p.add_argument('--max-runtime-seconds',type=int)
    p.add_argument('--conditions', nargs='+', default=['claim-only','wrong-42','wrong-1337','generic'])
    a = p.parse_args()
    allowed = ('claim-only','wrong-42','wrong-1337','generic') if a.study == 'grounding' else ('hybrid-1k','hybrid-2k','hybrid-4k','hybrid-8k','prefix-2k')
    if any(c not in allowed for c in a.conditions):
        raise ValueError('Unregistered control condition')
    inputs, out = Path(a.inputs), Path(a.output)
    manifest = json.loads((inputs/'manifest.json').read_text())
    assert manifest['model_revision'] == REVISION
    if a.study == 'context' and manifest.get('study') != 'registered_context_sweep':
        raise ValueError('Registered context input manifest required')
    head_path = Path(__file__).resolve().parents[1]/'deployment/head-v15.json'
    head = json.loads(head_path.read_text())
    assert head['model_revision'] == REVISION
    out.mkdir(parents=True,exist_ok=True)
    contract = {'input_manifest_sha256':sha(inputs/'manifest.json'),'head_sha256':sha(head_path),
                'engine_sha256':sha(Path(__file__).resolve().parents[1]/'src/ost_nli/v15.py'),
                'script_sha256':sha(__file__),'scope':'train/validation only; consumed310 untouched',
                'conditions':a.conditions,'device':a.device,'weights_frozen':True,'study':a.study}
    contract_path = out/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Resume source/input/head contract changed')
    contract_path.write_text(json.dumps(contract,indent=2)+'\n')
    began = time.perf_counter()
    engine = Engine(a.model_dir,device=a.device)
    metadata = {**engine.metadata,'model':engine.metadata['repo'],'model_revision':REVISION,
                'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'load_seconds':time.perf_counter()-began,**contract}
    (out/'experiment.json').write_text(json.dumps({**metadata,'status':'started'},indent=2)+'\n')
    for condition in a.conditions:
        directory = out/condition
        directory.mkdir(exist_ok=True)
        splits = ('train','validation') if condition == 'claim-only' or a.study == 'context' else ('validation',)
        for split in splits:
            entry = manifest['conditions'][f'{split}-{condition}']
            path = inputs/entry['file']
            if sha(path) != entry['sha256']:
                raise ValueError('Input file checksum mismatch')
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            chunks = directory/f'{split}-chunks'
            chunks.mkdir(exist_ok=True)
            preds, hidden, logits, ids = [], [], [], []
            for chunk in sorted(chunks.glob('[0-9][0-9][0-9][0-9][0-9][0-9].npz')):
                with np.load(chunk,allow_pickle=False) as saved:
                    pred = json.loads(saved['predictions'].item())
                    expected = rows[len(ids):len(ids)+len(pred)]
                    if saved['ids'].tolist() != [r['id'] for r in expected]:
                        raise ValueError('Chunk resume alignment failed')
                    if not np.isfinite(saved['hidden']).all() or not np.isfinite(saved['option_logits']).all():
                        raise ValueError('Nonfinite persisted features')
                    ids.extend(saved['ids'].tolist())
                    preds.extend(pred)
                    hidden.extend(saved['hidden'])
                    logits.extend(saved['option_logits'])
            pending = []
            for index in range(len(ids),len(rows)):
                if a.max_runtime_seconds and time.perf_counter()-began >= a.max_runtime_seconds:
                    raise TimeoutError('Registered encoding time budget reached; completed atomic chunks preserved')
                row = rows[index]
                # Only messages are passed: labels, donor IDs and references are not model inputs.
                result = engine.infer(row['messages'],method='head',head=head)
                h, l = result.pop('hidden'), result.pop('option_logits')
                if row.get('prompt_token_cap') is not None and result['context_tokens'] > row['prompt_token_cap']:
                    raise ValueError('Actual model prompt exceeds registered tokenizer budget')
                result.update(id=row['id'],context_bytes=row['context_bytes'],
                              latency_seconds=result['latency_seconds']+row['retrieval_seconds'],
                              truncated=row['truncated'],evidence_ids=row['evidence_ids'],
                              evidence_pages=row['evidence_pages'])
                pending.append((row['id'],h,l,result))
                if len(pending)==10 or index==len(rows)-1:
                    target = chunks/f'{len(preds):06d}.npz'
                    temporary = target.with_suffix('.partial.npz')
                    np.savez_compressed(temporary,ids=np.asarray([r[0] for r in pending]),
                        hidden=np.asarray([r[1] for r in pending],dtype=np.float32),
                        option_logits=np.asarray([r[2] for r in pending],dtype=np.float32),
                        predictions=np.asarray(json.dumps([r[3] for r in pending],allow_nan=False)))
                    temporary.replace(target)
                    ids.extend(r[0] for r in pending)
                    hidden.extend(r[1] for r in pending)
                    logits.extend(r[2] for r in pending)
                    preds.extend(r[3] for r in pending)
                    pending.clear()
                    print(f'PHASE2 {condition} {split} persisted={len(preds)}/{len(rows)} tokens={result["context_tokens"]}',flush=True)
            assert ids == [r['id'] for r in rows]
            np.savez_compressed(directory/f'{split}.npz',ids=np.asarray(ids),
                hidden=np.asarray(hidden,dtype=np.float32),option_logits=np.asarray(logits,dtype=np.float32),
                labels=np.asarray([r['label'] for r in rows],dtype=np.int64))
            (directory/f'{split}_rows.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in preds))
            metrics = evaluate([{**r,'evidence_ids':[]} for r in rows],preds)
            metrics.update(scoring_interpretation='Correct-booklet frozen full-trained-head context diagnostic; matched head fitting is separate' if a.study == 'context' else 'Original-label artifact retention ONLY; swapped pair labels unknown'
                if condition.startswith('wrong-') else 'Original-label frozen-head sensitivity control; empty evidence is a distribution shift')
            (directory/f'{split}-frozen-head-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        files = [{'path':path.name,'sha256':sha(path)} for path in sorted(directory.glob('*.npz'))]
        files.append({'path':'validation_rows.jsonl','sha256':sha(directory/'validation_rows.jsonl')})
        source = {**metadata,'context':condition,'status':'completed','files':files,
                  'train_sha256':manifest['conditions'][f'train-{condition}' if a.study == 'context' else 'train-claim-only']['dataset_sha256'],
                  'validation_sha256':manifest['conditions'][f'validation-{condition}']['dataset_sha256'],
                  'training':'Frozen Apertus encodings; any head fitting is a separate train-only CPU stage'}
        (directory/'experiment.json').write_text(json.dumps(source,indent=2)+'\n')
        print(f'PHASE2_CONTROL_COMPLETE {condition}',flush=True)
    (out/'experiment.json').write_text(json.dumps({**metadata,'status':'completed',
        'runtime_seconds':time.perf_counter()-began},indent=2)+'\n')
    print('PHASE2_CONTROLS_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
