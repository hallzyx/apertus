"""Verify frozen final validation; produce a decision without modifying deployment."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readlines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    import numpy as np
    from ost_nli.data import load_dataset,fingerprint,words
    from ost_nli.metrics import evaluate
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='track_2a/experiments/apertus-v15-phase2-final-validation-v1')
    p.add_argument('--output',default='track_2a/experiments/apertus-v15-phase2/final-results.json')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('Immutable final conclusion already exists')
    rows=load_dataset('track_2a/data/private/full-booklets-v2/validation.jsonl')
    assert fingerprint(rows)=='00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f'
    record=json.loads((root/'experiment.json').read_text())
    assert record['status']=='completed' and record['validation_sha256']==fingerprint(rows)
    assert record['training_performed'] is False and record['holdout_accessed'] is False
    for entry in record['files']:
        if sha(root/entry['path'])!=entry['sha256']:raise ValueError('Final recovery checksum mismatch')
    proposal=json.loads(Path('track_2a/experiments/apertus-v15-phase2/context-results.json').read_text())
    condition=proposal['provisional_choice'];assert record['context']==condition
    controlroot=Path('track_2a/experiments/apertus-v15-phase2-controls-v1')
    contract=json.loads((controlroot/'contract.json').read_text())
    originalhead=Path('track_2a/deployment/head-v15.json')
    assert sha(originalhead)==contract['head_sha256']
    headpath=Path(proposal['comparisons'][condition]['source_head']) if condition!='full' else originalhead
    assert sha(headpath)==record['head_sha256']
    head=json.loads(headpath.read_text());original=json.loads(originalhead.read_text())
    def verify_cache(path,predictions,selected_head):
        with np.load(path,allow_pickle=False) as cache:
            assert cache['ids'].tolist()==[r['id'] for r in rows]
            assert cache['labels'].tolist()==[r['label'] for r in rows]
            x=cache['hidden'].astype(np.float64)
            z=((x-np.asarray(selected_head['feature_mean']))/np.asarray(selected_head['feature_scale']))@np.asarray(selected_head['coefficients']).T+np.asarray(selected_head['intercept'])
            z/=selected_head['temperature'];q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
        aligned={p['id']:p for p in predictions}
        assert len(aligned)==len(rows)
        assert q.argmax(1).tolist()==[aligned[r['id']]['label'] for r in rows]
        assert np.allclose(q,[aligned[r['id']]['probabilities'] for r in rows],atol=1e-6)
        return q
    fresh=readlines(root/'correct-full-recheck.jsonl');verify_cache(root/'correct-full-recheck.npz',fresh,original)
    freshmetrics=evaluate(rows,fresh)
    base=[{**p,'label':p['native_base_scores_label'],'probabilities':p['native_base_scores_probabilities']} for p in fresh]
    historical=json.loads(Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/predictions.json').read_text())['predictions']
    old={p['id']:p for p in historical};current={p['id']:p for p in fresh}
    replication={'metrics':freshmetrics,'native_restricted_base_scores':evaluate(rows,base),
        'labels_changed_from_historical':sum(old[r['id']]['label']!=current[r['id']]['label'] for r in rows),
        'max_probability_difference':max(abs(a-b) for r in rows for a,b in zip(old[r['id']]['probabilities'],current[r['id']]['probabilities'])),
        'interpretation':'Same frozen head and original validation; native A6000 recheck versus historical A40. No refit.'}
    candidate=fresh if condition=='full' else json.loads((headpath.parent/'predictions.json').read_text())['predictions']
    candidate_metrics=evaluate(rows,candidate);candidate_by={p['id']:p for p in candidate};failures=[];controls={}
    for seed in (42,1337):
        if condition=='full':
            folder=controlroot/f'wrong-{seed}';pred=readlines(folder/'validation_rows.jsonl')
            source=json.loads((folder/'experiment.json').read_text())
            for entry in source['files']:assert sha(folder/entry['path'])==entry['sha256']
        else:
            folder=root/f'wrong-{seed}';pred=readlines(folder/'predictions.jsonl')
            verify_cache(folder/'validation.npz',pred,head)
            for r,prediction in zip(rows,pred):
                assert prediction['id']==r['id'] and prediction['donor_booklet_id']!=r['booklet_id']
                assert prediction['donor_language']==r['document_language']
        metrics=evaluate(rows,pred);f1=metrics['macro_f1'];baseline=candidate_metrics['macro_f1']
        controls[f'wrong-{seed}']={'original_label_retention_metrics':metrics,'retention':f1/baseline,'gap':baseline-f1}
        if f1/baseline>=.90 or baseline-f1<=.05:failures.append(f'wrong-{seed}_grounding_guard')
    emptyfolder=controlroot/'claim-only';manifest=json.loads((emptyfolder/'experiment.json').read_text())
    for entry in manifest['files']:assert sha(emptyfolder/entry['path'])==entry['sha256']
    with np.load(emptyfolder/'validation.npz',allow_pickle=False) as cache:
        assert cache['ids'].tolist()==[r['id'] for r in rows]
        x=cache['hidden'].astype(np.float64)
    z=((x-np.asarray(head['feature_mean']))/np.asarray(head['feature_scale']))@np.asarray(head['coefficients']).T+np.asarray(head['intercept'])
    z/=head['temperature'];q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
    empty=[{'id':r['id'],'label':int(p.argmax()),'probabilities':p.tolist()} for r,p in zip(rows,q)]
    metrics=evaluate(rows,empty);f1=metrics['macro_f1'];baseline=candidate_metrics['macro_f1']
    controls['empty']={'original_label_retention_metrics':metrics,'retention':f1/baseline,'gap':baseline-f1,
        'scope':'Cached actual claim-only encoder features, frozen candidate head; no new head fitting'}
    if f1/baseline>=.90 or baseline-f1<=.05:failures.append('empty_grounding_guard')
    if condition!='full':
        # Apply the same predeclared guards to the fresh full recheck as well.
        if baseline<freshmetrics['macro_f1']-.02:failures.append('fresh_full_macro_guard')
        if candidate_metrics['cross_lingual']['macro_f1']<freshmetrics['cross_lingual']['macro_f1']-.03:failures.append('fresh_full_cross_lingual_guard')
        if any(candidate_metrics['per_class'][str(c)]['f1']<freshmetrics['per_class'][str(c)]['f1']-.05 for c in range(3)):failures.append('fresh_full_class_guard')
    cli=json.loads((root/'cli-integration.json').read_text());front=json.loads((root/'frontend-real-inference.json').read_text())
    assert len(cli)==3 and {(p['claim_language'],p['document_language']) for p in cli}=={('de','fr'),('fr','it'),('it','de')}
    if not all(p['label_reproduced'] for p in cli) or not front['label_reproduced']:failures.append('real_pdf_api_replication')
    selected=condition if not failures else 'full'
    chosen=candidate if selected==condition else fresh;by={p['id']:p for p in chosen}
    seen=set();unique=[]
    for r in rows:
        key=(r['document_id'],' '.join(words(r['claim'])))
        if key not in seen:seen.add(key);unique.append(r)
    result={'status':'completed_frozen_final_validation','provisional_choice':condition,'final_choice':selected,
        'candidate_head_sha256':sha(headpath),'candidate_metrics':candidate_metrics,'guard_failures':failures,
        'full_recheck':replication,'context_matched_controls':controls,
        'final_validation_metrics':evaluate(rows,chosen),'deduplicated_macro_f1':evaluate(unique,[by[r['id']] for r in unique])['macro_f1'],
        'by_voting_event':{event:evaluate([r for r in rows if r['booklet_id']==event],[by[r['id']] for r in rows if r['booklet_id']==event]) for event in sorted({r['booklet_id'] for r in rows})},
        'real_pdf_cli':cli,'real_frontend':front,'actual_gpu_forward_count':record['gpu_forward_count'],
        'training_performed':False,'lora_performed':False,'consumed310_accessed':False,'deployment_changed':False,
        'limitations':'Architecture selected on276 validation,207 distinct pairs,3events/2connected units. Wrong/empty original labels measure retention, not intervention NLI accuracy. No new held-out performance claim. Evidence provenance is not sufficiency.',
        'source_experiment_sha256':sha(root/'experiment.json')}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'candidate':condition,'final':selected,'failures':failures,'f1':result['final_validation_metrics']['macro_f1'],'controls':{k:v['original_label_retention_metrics']['macro_f1'] for k,v in controls.items()}},indent=2))


if __name__=='__main__':main()
