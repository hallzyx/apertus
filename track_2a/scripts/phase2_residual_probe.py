"""Conditional train-only cached-feature artifact investigation; research-only heads."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import numpy as np
    from ost_nli.data import load_dataset, fingerprint
    from ost_nli.metrics import evaluate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls',default='track_2a/experiments/apertus-v15-phase2-controls-v1')
    p.add_argument('--output',default='track_2a/experiments/apertus-v15-phase2-residual-v1')
    a = p.parse_args()
    gate = json.loads(Path('track_2a/experiments/apertus-v15-phase2/grounding-results.json').read_text())
    if gate['grounding_gate'] != 'WARNING_PAUSE_ARCHITECTURE_OPTIMIZATION':
        raise ValueError('Residual probe is registered only for artifact-warning investigation')
    controls = Path(a.controls)
    full = Path('track_2a/experiments/apertus-v15-research-v1/feature-cache')
    blank = controls/'claim-only'
    for directory in (full,blank):
        record = json.loads((directory/'experiment.json').read_text())
        if record['status'] != 'completed' or record['model_revision'] != 'a411d838600baf0e3635a3daf66fb7c55fc97bb6':
            raise ValueError('Completed original-model caches required')
        for entry in record['files']:
            if checksum(directory/entry['path']) != entry['sha256']:
                raise ValueError('Cached input checksum mismatch')
    out = Path(a.output)
    if out.exists():
        raise ValueError('Immutable residual investigation exists')
    out.mkdir(parents=True)
    cache = out/'feature-cache'
    cache.mkdir()
    datasets = {s:load_dataset(f'track_2a/data/private/full-booklets-v2/{s}.jsonl') for s in ('train','validation')}
    blank_features = {}
    for split in ('train','validation'):
        with np.load(full/f'{split}.npz',allow_pickle=False) as f,np.load(blank/f'{split}.npz',allow_pickle=False) as b:
            rows = datasets[split]
            assert f['ids'].tolist() == b['ids'].tolist() == [r['id'] for r in rows]
            assert f['labels'].tolist() == b['labels'].tolist() == [r['label'] for r in rows]
            delta = {name:f[name]-b[name] for name in ('hidden','option_logits')}
            if not all(np.isfinite(v).all() for v in delta.values()):
                raise ValueError('Nonfinite residual features')
            np.savez_compressed(cache/f'{split}.npz',ids=f['ids'],labels=f['labels'],**delta)
            blank_features[split] = {name:b[name].astype(np.float64) for name in ('hidden','option_logits')}
    full_metadata = [json.loads(s) for s in (full/'validation_rows.jsonl').read_text().splitlines()]
    blank_metadata = [json.loads(s) for s in (blank/'validation_rows.jsonl').read_text().splitlines()]
    assert [r['id'] for r in full_metadata] == [r['id'] for r in blank_metadata]
    encoded = [{**f,'context_tokens':f['context_tokens']+b['context_tokens'],
                'latency_seconds':f['latency_seconds']+b['latency_seconds'],
                'encoding_note':'Sum of historical A40 full and new A6000 blank measurements; not a paired live latency benchmark'} for f,b in zip(full_metadata,blank_metadata)]
    (cache/'validation_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in encoded))
    source = json.loads((full/'experiment.json').read_text())
    source.update(context='full-minus-claim',files=[{'path':p.name,'sha256':checksum(p)} for p in sorted(cache.iterdir()) if p.is_file()],
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_script_sha256=checksum(__file__),
                  train_sha256=fingerprint(datasets['train']),validation_sha256=fingerprint(datasets['validation']),
                  feature_definition='Joint hidden/option logits minus claim-only equivalents; same original frozen Apertus weights',
                  source_full_manifest_sha256=checksum(full/'experiment.json'),source_blank_manifest_sha256=checksum(blank/'experiment.json'),
                  latency_note='Cross-GPU cached sum, not a new integrated inference measurement')
    (cache/'experiment.json').write_text(json.dumps(source,indent=2)+'\n')
    results = {}
    for feature in ('hidden','option_logits'):
        directory = out/feature
        subprocess.run([str(Path(__import__('sys').executable)),'track_2a/scripts/train_v15_head.py',
            '--cache',str(cache),'--context','full-minus-claim','--features',feature,
            '--train','track_2a/data/private/full-booklets-v2/train.jsonl',
            '--validation','track_2a/data/private/full-booklets-v2/validation.jsonl',
            '--output-dir',str(directory)],check=True)
        head = json.loads((directory/'head.json').read_text())
        kind = 'paired_hidden_difference' if feature=='hidden' else 'paired_option_logits_difference'
        # These research parameters must not accidentally be served by a single-forward Engine.
        head.update(feature_kind=kind,paired_input_required=True,production_runtime_compatible=False)
        (directory/'head.json').write_text(json.dumps(head,indent=2)+'\n')
        metrics = json.loads((directory/'metrics.json').read_text())
        conditions = {}
        for condition in ('claim-only','wrong-42','wrong-1337','generic'):
            folder = controls/condition
            original_manifest = json.loads((folder/'experiment.json').read_text())
            for entry in original_manifest['files']:
                if checksum(folder/entry['path']) != entry['sha256']:
                    raise ValueError('Intervention feature checksum mismatch')
            with np.load(folder/'validation.npz',allow_pickle=False) as values:
                assert values['ids'].tolist() == [r['id'] for r in datasets['validation']]
                x = values[feature].astype(np.float64)-blank_features['validation'][feature]
            if feature=='option_logits':
                x -= x.mean(1,keepdims=True)
            z = ((x-np.asarray(head['feature_mean']))/np.asarray(head['feature_scale']))@np.asarray(head['coefficients']).T+np.asarray(head['intercept'])
            q = np.exp(z/head['temperature']-(z/head['temperature']).max(1,keepdims=True))
            q /= q.sum(1,keepdims=True)
            pred = [{'id':r['id'],'label':int(p.argmax()),'probabilities':p.tolist()} for r,p in zip(datasets['validation'],q)]
            conditions[condition] = evaluate(datasets['validation'],pred)
            (directory/f'{condition}-predictions.json').write_text(json.dumps(pred,indent=2,allow_nan=False)+'\n')
        results[feature] = {'correct_context':metrics,'interventions':conditions,
            'head_sha256':checksum(directory/'head.json'),'feature_definition':kind,
            'warning_conditions':[c for c,m in conditions.items() if c.startswith('wrong-') and
                (m['macro_f1'] >= metrics['macro_f1']-.05 or m['macro_f1']/metrics['macro_f1'] >= .90)],
            'caution':'Claim-only residual is identically zero by construction. Its poor performance is tautological, not grounding evidence; wrong-document interventions are the substantive control. Swapped-pair labels remain unknown.'}
        print('RESIDUAL_PROBE',feature,metrics['macro_f1'],{c:m['macro_f1'] for c,m in conditions.items()},flush=True)
    (out/'investigation.json').write_text(json.dumps({'status':'completed_research_only','trigger':gate['grounding_gate'],
        'fit_scope':'Training902 only, original grouped OOF C and calibration; validation architecture diagnostics',
        'holdout_used':False,'new_gpu_forwards':0,'additional_compute_cost_usd':0,
        'production_selection_changed':False,'source_script_sha256':checksum(__file__),'results':results},indent=2)+'\n')


if __name__ == '__main__':
    main()
