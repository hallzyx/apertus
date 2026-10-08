"""Recompute registered context comparisons and propose, never deploy, a candidate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    from ost_nli.data import load_dataset,fingerprint,words
    from ost_nli.metrics import evaluate,classification
    from phase2_cpu_audit import connected_groups
    from ost_nli.documents import reference_coverage
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='track_2a/experiments/apertus-v15-phase2-context-v1')
    p.add_argument('--output',default='track_2a/experiments/apertus-v15-phase2/context-results.json')
    p.add_argument('--references',default='track_2a/data/private/splits-strict/validation.jsonl',help='Development-only reference adapter; never model input')
    a=p.parse_args();root=Path(a.root);out=Path(a.output)
    if out.exists():raise ValueError('Immutable context conclusion already exists')
    rows=load_dataset('track_2a/data/private/full-booklets-v2/validation.jsonl')
    assert fingerprint(rows)=='00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f'
    refs={r['id']:r for r in load_dataset(a.references)}
    assert set(refs)=={r['id'] for r in rows} and all(refs[r['id']]['label']==r['label'] for r in rows)
    assert json.loads((root/'experiment.json').read_text())['status']=='completed'
    contract=json.loads((root/'contract.json').read_text())
    assert hashlib.sha256((root/'inputs_manifest.json').read_bytes()).hexdigest()==contract['input_manifest_sha256']
    protocol=json.loads(Path('track_2a/experiments/apertus-v15-phase2/context-protocol.json').read_text())
    baseline_path=Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/predictions.json')
    baseline=json.loads(baseline_path.read_text())['predictions'];full=evaluate(rows,baseline)
    reference=json.loads((root/'inputs_manifest.json').read_text())
    seen=set();unique=[]
    for row in rows:
        k=(row['document_id'],' '.join(words(row['claim'])))
        if k not in seen:seen.add(k);unique.append(row)
    comparisons={};candidates=[];predictions={}
    for condition in protocol['conditions']:
        directory=root/condition;source=json.loads((directory/'experiment.json').read_text())
        if source['status']!='completed' or source['validation_sha256']!=fingerprint(rows):raise ValueError('Incomplete or mismatched feature cache')
        for entry in source['files']:
            if hashlib.sha256((directory/entry['path']).read_bytes()).hexdigest()!=entry['sha256']:
                raise ValueError('Context cache checksum mismatch')
        fitted=root/(condition+'-fitted-head')
        audit=json.loads((fitted/'training_audit.json').read_text());assert audit['temperature_fit_uses_validation'] is False
        record=json.loads((fitted/'predictions.json').read_text());assert record['dataset_sha256']==fingerprint(rows)
        pred=record['predictions'];metrics=evaluate(rows,pred);predictions[condition]=pred
        head=json.loads((fitted/'head.json').read_text());assert head['context']==condition and head['class_order']==[0,1,2]
        with np.load(directory/'validation.npz',allow_pickle=False) as cache:
            assert cache['ids'].tolist()==[r['id'] for r in rows]
            assert cache['labels'].tolist()==[r['label'] for r in rows]
            x=cache['hidden'].astype(np.float64);scores=cache['option_logits'].astype(np.float64)
        z=((x-np.asarray(head['feature_mean']))/np.asarray(head['feature_scale']))@np.asarray(head['coefficients']).T+np.asarray(head['intercept'])
        q=np.exp(z/head['temperature']-(z/head['temperature']).max(1,keepdims=True));q/=q.sum(1,keepdims=True)
        aligned={p['id']:p for p in pred}
        assert np.argmax(q,axis=1).tolist()==[aligned[r['id']]['label'] for r in rows]
        assert np.allclose(q,np.asarray([aligned[r['id']]['probabilities'] for r in rows]),atol=1e-10)
        qbase=np.exp(scores-scores.max(1,keepdims=True));qbase/=qbase.sum(1,keepdims=True)
        base=[{**aligned[r['id']],'label':int(v.argmax()),'probabilities':v.tolist()} for r,v in zip(rows,qbase)]
        unchanged=[json.loads(s) for s in (directory/'validation_rows.jsonl').read_text().splitlines()]
        evidence=[json.loads(s) for s in (directory/'validation-evidence.jsonl').read_text().splitlines()]
        assert [p['id'] for p in evidence]==[r['id'] for r in rows]
        checked_quotes=0
        for row,record in zip(rows,evidence):
            passages={p['id']:p for p in row['passages']}
            for passage in record['selected_passages']:
                original=passages[passage['id']]
                for key in ('text','page','char_start','char_end','source_sha256'):
                    assert passage.get(key)==original.get(key)
                checked_quotes+=1
        cap=protocol['conditions'][condition]['prompt_cap']
        assert all(p['context_tokens']<=cap for p in pred)
        failures=[]
        if metrics['macro_f1']<full['macro_f1']-.02:failures.append('macro_f1_gap_exceeds0.02')
        if metrics['cross_lingual']['macro_f1']<full['cross_lingual']['macro_f1']-.03:failures.append('cross_lingual_gap_exceeds0.03')
        if any(metrics['per_class'][str(c)]['f1']<full['per_class'][str(c)]['f1']-.05 for c in range(3)):failures.append('class_f1_gap_exceeds0.05')
        coverages=[reference_coverage(record['selected_passages'],'\n\n'.join(p['text'] for p in refs[row['id']]['passages'])) for row,record in zip(rows,evidence)]
        observed=[v for v in coverages if v is not None]
        comparisons[condition]={'matched_head':metrics,'native_restricted_base_scores':evaluate(rows,base),
            'unchanged_full_head':evaluate(rows,unchanged),'deduplicated_macro_f1':evaluate(unique,[aligned[r['id']] for r in unique])['macro_f1'],
            'head_sha256':hashlib.sha256((fitted/'head.json').read_bytes()).hexdigest(),'source_head':str(fitted/'head.json'),
            'registered_selection_failures':failures,'token_cap':cap,
            'reference_5gram_coverage':float(np.mean(observed)) if observed else None,
            'reference_coverage_definition':'Same page-local dehyphenated5gram diagnostic as original retrieval audit; no cross-passage ngrams',
            'preparation_raw_overlap':reference['conditions'][f'validation-{condition}']['mean_reference_5gram_coverage'],
            'mean_selected_passages':sum(len(p['selected_passages']) for p in evidence)/len(evidence),
            'exact_source_quote_page_offset_checks':checked_quotes,
            'scope_caution':'Candidate source passages preserve provenance; lexical overlap is not semantic evidence scoring'}
        if not failures:candidates.append((metrics['average_context_tokens'],-metrics['macro_f1'],condition))
    choice=min(candidates)[2] if candidates else 'full'
    groups=np.asarray(connected_groups(rows));units=sorted(set(groups.tolist()))
    generator=np.random.default_rng(42)
    original_labels={p['id']:p['label'] for p in baseline}
    deltas={condition:[] for condition in predictions}
    for _ in range(2000):
        selected=np.concatenate([np.flatnonzero(groups==unit) for unit in generator.choice(units,len(units),replace=True)])
        truth=[rows[int(i)]['label'] for i in selected]
        original=classification(truth,[original_labels[rows[int(i)]['id']] for i in selected])['macro_f1']
        for condition,pred in predictions.items():
            aligned={p['id']:p['label'] for p in pred}
            value=classification(truth,[aligned[rows[int(i)]['id']] for i in selected])['macro_f1']
            deltas[condition].append(value-original)
    resampling={'seed':42,'replicates':2000,'connected_units':len(units),
        'unit_sizes':[int((groups==unit).sum()) for unit in units],
        'paired_f1_difference_percentiles_2_5_97_5':{condition:np.quantile(values,[.025,.975]).tolist() for condition,values in deltas.items()},
        'interpretation':'Descriptive sensitivity only: two duplicate-connected units cannot support reliable population confidence intervals or superiority claims. Not used for selection.'}
    simulations=[]
    if choice!='full':
        compact={p['id']:p for p in predictions[choice]};original={p['id']:p for p in baseline}
        for threshold in [.5,.7,.85,.95]:
            routed=[];escalated=0
            for row in rows:
                p=compact[row['id']];fallback=max(p['probabilities'])<threshold
                if fallback:
                    escalated+=1;p={**original[row['id']],
                        'context_tokens':p['context_tokens']+original[row['id']]['context_tokens'],
                        'latency_seconds':p['latency_seconds']+original[row['id']]['latency_seconds']}
                routed.append(p)
            simulations.append({'threshold':threshold,'escalated':escalated,'metrics':evaluate(rows,routed)})
    points={condition:(value['matched_head']['macro_f1'],value['matched_head']['average_context_tokens'],value['matched_head']['average_latency_seconds']) for condition,value in comparisons.items()}
    points['historical-full']=(full['macro_f1'],full['average_context_tokens'],full['average_latency_seconds'])
    frontier=[name for name,(f1,tokens,seconds) in points.items() if not any(
        other!=name and score>=f1 and count<=tokens and latency<=seconds and
        (score>f1 or count<tokens or latency<seconds)
        for other,(score,count,latency) in points.items())]
    result={'status':'completed_context_comparison_provisional_selection','scope':'Validation276 architecture development only; no consumed310 access',
        'original_full':full,'comparisons':comparisons,'provisional_choice':choice,
        'deployment_changed':False,'requires_context_matched_grounding_check':choice!='full',
        'decision_rule':protocol['selection_rule'],'per_class_guard':protocol['per_class_guard'],
        'group_aware_paired_resampling':resampling,
        'observed_f1_token_time_frontier':frontier,
        'frontier_caution':'Observed cached timing tradeoff, not paired live service latency; historical-full A40 versus new A6000. Oracle excluded. Selection additionally requires registered class/language/grounding guards.',
        'adaptive_routing_simulations':simulations,'adaptive_routing_selected':False,
        'routing_caution':'Fixed-grid cached simulations, no trained router or threshold adopted. Escalation costs both forwards. Compact on A6000 plus historical full on A40 is not paired live latency.',
        'validation_cautions':'Three events, two duplicate-connected units,207 unique claim/document pairs; reference-vs-booklet scope conflict remains. High validation F1 alone does not establish generalization.',
        'lora_performed':False,'protocol_sha256':hashlib.sha256(Path('track_2a/experiments/apertus-v15-phase2/context-protocol.json').read_bytes()).hexdigest(),
        'actual_source_contract':contract,'baseline_prediction_sha256':hashlib.sha256(baseline_path.read_bytes()).hexdigest()}
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'choice':choice,'results':{k:{'f1':v['matched_head']['macro_f1'],'tokens':v['matched_head']['average_context_tokens'],'failures':v['registered_selection_failures']} for k,v in comparisons.items()}},indent=2))


if __name__=='__main__':main()
