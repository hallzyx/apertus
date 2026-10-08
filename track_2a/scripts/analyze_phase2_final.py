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
    extension=record.get('accuracy_extension',False)
    condition='hybrid-8k' if extension else proposal['provisional_choice'];assert record['context']==condition
    prior_root=Path('track_2a/experiments/apertus-v15-phase2-final-validation-v1')
    prior=None
    if extension:
        prior=json.loads(Path('track_2a/experiments/apertus-v15-phase2/final-results.json').read_text())
        assert prior['final_choice']=='hybrid-4k' and not prior['guard_failures']
        assert sha(prior_root/'experiment.json')==prior['source_experiment_sha256']==record['reused_full_source_experiment_sha256']
        for entry in json.loads((prior_root/'experiment.json').read_text())['files']:
            assert sha(prior_root/entry['path'])==entry['sha256']
        for name in ('correct-full-recheck.jsonl','correct-full-recheck-metrics.json','neighbors-diagnostic-predictions.jsonl','diverse-diagnostic-predictions.jsonl'):
            assert sha(root/name)==sha(prior_root/name)
    controlroot=Path('track_2a/experiments/apertus-v15-phase2-controls-v1')
    contract=json.loads((controlroot/'contract.json').read_text())
    originalhead=Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/head.json')
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
    fresh=readlines(root/'correct-full-recheck.jsonl');verify_cache((prior_root if extension else root)/'correct-full-recheck.npz',fresh,original)
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
        neutral=[p for p in pred if p['label']==1]
        controls[f'wrong-{seed}']={'original_label_retention_metrics':metrics,'retention':f1/baseline,'gap':baseline-f1,
            'neutral_prediction_rate':len(neutral)/len(pred),
            'mean_neutral_confidence':sum(p['probabilities'][1] for p in neutral)/len(neutral) if neutral else None,
            'prediction_change_rate':sum(p['label']!=candidate_by[p['id']]['label'] for p in pred)/len(pred),
            'calibration_caution':'Probabilities against original labels here do not assess true swapped-premise NLI calibration'}
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
    if extension:
        previous=prior['final_validation_metrics']
        if baseline<previous['macro_f1']+.04:failures.append('accuracy_material_macro_gain')
        if candidate_metrics['cross_lingual']['macro_f1']<previous['cross_lingual']['macro_f1']+.03:failures.append('accuracy_material_cross_gain')
        if any(candidate_metrics['per_class'][str(c)]['f1']<previous['per_class'][str(c)]['f1'] for c in range(3)):failures.append('accuracy_class_regression')
        if candidate_metrics['average_context_tokens']>freshmetrics['average_context_tokens']*.75:failures.append('accuracy_context_efficiency')
        if any(candidate_metrics['calibration'][k]>previous['calibration'][k] for k in ('brier_score','negative_log_likelihood')):failures.append('accuracy_calibration_regression')
    cli=json.loads((root/'cli-integration.json').read_text());front=json.loads((root/'frontend-real-inference.json').read_text())
    assert len(cli)==3 and {(p['claim_language'],p['document_language']) for p in cli}=={('de','fr'),('fr','it'),('it','de')}
    if not all(p['label_reproduced'] for p in cli) or not front['label_reproduced']:failures.append('real_pdf_api_replication')
    diagnostics=json.loads((root/'bounded-context-evidence-diagnostics.json').read_text())
    sample=[]
    for language in ('de','fr','it'):
        for label in (0,1,2):sample.extend([r for r in rows if r['claim_language']==language and r['label']==label][:10])
    assert len(sample)==90
    head4path=Path(proposal['comparisons']['hybrid-4k']['source_head']);head4=json.loads(head4path.read_text())
    assert sha(head4path)==proposal['comparisons']['hybrid-4k']['head_sha256']
    diagnostic_predictions={name:readlines(root/f'{name}-diagnostic-predictions.jsonl') for name in ('neighbors','diverse')}
    for name,values in diagnostic_predictions.items():
        measured=evaluate(sample,values)
        assert abs(measured['macro_f1']-diagnostics['neighbors_same90' if name=='neighbors' else 'diversity_same90']['macro_f1'])<1e-12
        assert all(p['context_tokens']<=4096 for p in values)
    outputs={(name,p['id']):p for name,values in diagnostic_predictions.items() for p in values}
    outputs.update({('evidence-only',p['id']):p['output'] for p in diagnostics['evidence_proofs']})
    from ost_nli.model import select_context,nli_messages
    rows_by={r['id']:r for r in rows}
    for (_,iid),output in outputs.items():
        row=rows_by[iid];lookup={p['id']:p for p in row['passages']}
        passages=[lookup[pid] for pid in output['selected_passage_ids']]
        _,context=select_context({'passages':passages},mode='full',max_bytes=1000000)
        expected=hashlib.sha256(json.dumps(nli_messages(context,row['claim']),ensure_ascii=False).encode()).hexdigest()
        assert expected==output['prompt_sha256']
    for proof in diagnostics['evidence_proofs']:
        lookup={p['id']:p for p in rows_by[proof['id']]['passages']}
        for passage in proof['proposed_evidence']:
            assert all(passage.get(k)==lookup[passage['id']].get(k) for k in ('text','page','char_start','char_end','source_sha256'))
    new_outputs={key:value for key,value in outputs.items() if not extension or key[0]=='evidence-only'}
    with np.load(root/'bounded-diagnostic-features.npz',allow_pickle=False) as cache:
        assert len(cache['ids'])==len(new_outputs)==diagnostics['gpu_forward_count']
        for iid,mode,x in zip(cache['ids'].tolist(),cache['modes'].tolist(),cache['hidden']):
            h=head if mode=='evidence-only' else head4
            z=((x.astype(np.float64)-np.asarray(h['feature_mean']))/np.asarray(h['feature_scale']))@np.asarray(h['coefficients']).T+np.asarray(h['intercept'])
            z/=h['temperature'];q=np.exp(z-z.max());q/=q.sum()
            output=outputs[(mode,iid)]
            assert int(q.argmax())==output['label'] and np.allclose(q,output['probabilities'],atol=1e-6)
    assert record['gpu_forward_count']==(556 if extension else (832 if condition!='full' else 280))+diagnostics['gpu_forward_count']
    selected=condition if not failures else ('hybrid-4k' if extension else 'full')
    chosen=candidate if selected==condition else (json.loads(Path(proposal['comparisons']['hybrid-4k']['source_head']).with_name('predictions.json').read_text())['predictions'] if extension else fresh);by={p['id']:p for p in chosen}
    if selected=='full':selected_base={p['id']:p for p in base}
    else:
        directory=Path('track_2a/experiments/apertus-v15-phase2-context-v1')/selected
        with np.load(directory/'validation.npz',allow_pickle=False) as cache:
            assert cache['ids'].tolist()==[r['id'] for r in rows]
            scores=cache['option_logits'].astype(np.float64)
        q=np.exp(scores-scores.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
        selected_base={r['id']:{'id':r['id'],'label':int(p.argmax()),'probabilities':p.tolist()} for r,p in zip(rows,q)}
    seen=set();unique=[]
    for r in rows:
        key=(r['document_id'],' '.join(words(r['claim'])))
        if key not in seen:seen.add(key);unique.append(r)
    buckets=[]
    for low,high in ((0,.5),(.5,.7),(.7,.85),(.85,.95),(.95,1.000001)):
        subset=[r for r in rows if low<=max(by[r['id']]['probabilities'])<high]
        buckets.append({'low':low,'high':min(high,1),'n':len(subset),
            'accuracy':sum(by[r['id']]['label']==r['label'] for r in subset)/len(subset) if subset else None,
            'mean_confidence':sum(max(by[r['id']]['probabilities']) for r in subset)/len(subset) if subset else None})
    original_errors={p['id']:p for p in readlines(Path('track_2a/experiments/apertus-v15-phase2/final-error-analysis.jsonl'))}
    errors=[{'id':r['id'],'gold':r['label'],'predicted':by[r['id']]['label'],
        'claim_language':r['claim_language'],'document_language':r['document_language'],
        'booklet_id':r['booklet_id'],'claim':r['claim'],
        'base_restricted_label_same_context':selected_base[r['id']]['label'],
        'base_correct_head_wrong_same_context':selected_base[r['id']]['label']==r['label'],
        'original_source_inspection':{k:v for k,v in original_errors.get(r['id'],{}).items()
            if k in ('suspected_failure_mode','notes','diagnostic_comparison')},
        'cause':'Undetermined; original inspected hypothesis, if present, is not proof of cause for the selected context'}
        for r in rows if by[r['id']]['label']!=r['label']]
    result={'accuracy_extension':extension,'architecture_selection_scope':'Exploratory accuracy extension after immutable registered 4k efficiency decision' if extension else 'Registered efficiency selection','status':'completed_frozen_final_validation','provisional_choice':condition,'final_choice':selected,
        'candidate_head_sha256':sha(headpath),'candidate_metrics':candidate_metrics,'guard_failures':failures,
        'full_recheck':replication,'context_matched_controls':controls,
        'final_validation_metrics':evaluate(rows,chosen),'deduplicated_macro_f1':evaluate(unique,[by[r['id']] for r in unique])['macro_f1'],
        'confidence_buckets':buckets,'final_validation_errors':errors,
        'selected_context_base_metrics':evaluate(rows,list(selected_base.values())),
        'selected_base_head_disagreement':{
            'both_correct':sum(by[r['id']]['label']==r['label'] and selected_base[r['id']]['label']==r['label'] for r in rows),
            'head_correct_base_wrong':sum(by[r['id']]['label']==r['label'] and selected_base[r['id']]['label']!=r['label'] for r in rows),
            'base_correct_head_wrong':sum(by[r['id']]['label']!=r['label'] and selected_base[r['id']]['label']==r['label'] for r in rows),
            'both_wrong':sum(by[r['id']]['label']!=r['label'] and selected_base[r['id']]['label']!=r['label'] for r in rows)},
        'bounded_neighbor_diversity_evidence_diagnostics':diagnostics,
        'by_voting_event':{event:evaluate([r for r in rows if r['booklet_id']==event],[by[r['id']] for r in rows if r['booklet_id']==event]) for event in sorted({r['booklet_id'] for r in rows})},
        'real_pdf_cli':cli,'real_frontend':front,'actual_gpu_forward_count':record['gpu_forward_count'],
        'training_performed':False,'lora_performed':False,'consumed310_accessed':False,'deployment_changed':False,
        'limitations':'Architecture selected on276 validation,207 distinct pairs,3events/2connected units. Wrong/empty original labels measure retention, not intervention NLI accuracy. No new held-out performance claim. Evidence provenance is not sufficiency.',
        'source_experiment_sha256':sha(root/'experiment.json')}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'candidate':condition,'final':selected,'failures':failures,'f1':result['final_validation_metrics']['macro_f1'],'controls':{k:v['original_label_retention_metrics']['macro_f1'] for k,v in controls.items()}},indent=2))


if __name__=='__main__':main()
