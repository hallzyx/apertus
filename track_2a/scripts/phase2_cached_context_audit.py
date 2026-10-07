"""Frozen full-head interventions on actual cached validation encodings, no fitting."""
import json
import hashlib
from pathlib import Path


def main():
    import numpy as np
    from ost_nli.data import load_dataset
    from ost_nli.metrics import evaluate
    rows = load_dataset('track_2a/data/private/full-booklets-v2/validation.jsonl')
    root = Path('track_2a/experiments/apertus-v15-research-v1')
    head = json.loads(Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/head.json').read_text())
    recorded = json.loads((root/'experiment.json').read_text())
    expected = {r['path']:r['sha256'] for r in recorded['files']}
    original = json.loads((root/'apertus-v15-full-hidden-v1/predictions.json').read_text())['predictions']
    original = {r['id']:r for r in original}
    output = Path('track_2a/experiments/apertus-v15-phase2/cached-context-audit')
    if output.exists():
        raise ValueError('Immutable experiment output exists')
    output.mkdir()
    results = {}
    for mode in ('full','bm25','dense','hybrid','reference'):
        folder = root/'validation'/f'{mode}-score'
        for name in ('features.npz','predictions.jsonl'):
            path = folder/name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected[str(path.relative_to(root))]:
                raise ValueError('Original recovered feature/prediction checksum mismatch')
        with np.load(folder/'features.npz',allow_pickle=False) as cache:
            assert cache['ids'].tolist() == [r['id'] for r in rows]
            x = cache['hidden'].astype(np.float64)
        z = (((x-np.asarray(head['feature_mean']))/np.asarray(head['feature_scale']))@np.asarray(head['coefficients']).T+np.asarray(head['intercept']))/head['temperature']
        q = np.exp(z-z.max(axis=1,keepdims=True))
        q /= q.sum(axis=1,keepdims=True)
        encoded = [json.loads(s) for s in (folder/'predictions.jsonl').read_text().splitlines()]
        assert [r['id'] for r in encoded] == [r['id'] for r in rows]
        pred = [{**e,'label':int(p.argmax()),'probabilities':p.tolist()} for e,p in zip(encoded,q)]
        if mode == 'full' and [p['label'] for p in pred] != [original[r['id']]['label'] for r in rows]:
            raise ValueError('Frozen full-head cached reproduction mismatch')
        metrics = evaluate(rows,pred)
        material = [i for i,r in enumerate(rows) if original[r['id']]['label'] in (0,2)]
        preserved = sum(pred[i]['label']==original[rows[i]['id']]['label'] for i in material)
        results[mode] = {'metrics':metrics,'head':'Same full-context frozen head; not a condition-specific refit',
                         'distribution_shift_caution':'Changes may reflect head covariate shift, missing evidence, or both',
                         'entailment_contradiction_decision_preservation':preserved/len(material),
                         'n_material_predictions':len(material),
                         'scope':'Reference is oracle diagnostic only; retrieval passages are candidate evidence, not established minimal proof',
                         'context_tokens_p50_p95':np.quantile([e['context_tokens'] for e in encoded],[.5,.95]).tolist(),
                         'encoding_latency_p50_p95':np.quantile([e['latency_seconds'] for e in encoded],[.5,.95]).tolist()}
        (output/f'{mode}-predictions.json').write_text(json.dumps(pred,indent=2,allow_nan=False)+'\n')
        print(mode,metrics['macro_f1'],results[mode]['entailment_contradiction_decision_preservation'],flush=True)
    (output/'audit.json').write_text(json.dumps({'scope':'validation-only frozen-head cached-feature intervention',
        'actual_new_gpu_forwards':0,'cost_usd':0,'results':results},indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
