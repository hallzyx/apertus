"""Bounded, preregistered neighboring/diversity and evidence-only diagnostics."""
import hashlib
import json
import time
from pathlib import Path


def run_diagnostics(engine,retriever,rows,head,condition,candidate_predictions,root):
    from ost_nli.context_budget import CONDITIONS,pack_passages
    from ost_nli.model import select_context,nli_messages
    from ost_nli.data import words
    from ost_nli.metrics import evaluate
    import numpy as np
    proposal=json.loads(Path('track_2a/experiments/apertus-v15-phase2/context-results.json').read_text())
    head4_path=Path(proposal['comparisons']['hybrid-4k']['source_head'])
    assert hashlib.sha256(head4_path.read_bytes()).hexdigest()==proposal['comparisons']['hybrid-4k']['head_sha256']
    head4=json.loads(head4_path.read_text())
    baseline4=json.loads((head4_path.parent/'predictions.json').read_text())['predictions']
    baseline_by={p['id']:p for p in baseline4};candidate_by={p['id']:p for p in candidate_predictions}
    sample=[]
    for lang in ('de','fr','it'):
        for label in (0,1,2):
            subset=[r for r in rows if r['claim_language']==lang and r['label']==label]
            assert len(subset)>=10
            sample.extend(subset[:10])
    features=[]
    def inferred(row,passages,selected_head,cap,mode):
        began=time.perf_counter()
        if cap:selected,context,_=pack_passages(engine.tokenizer,passages,row['claim'],cap)
        else:selected,context=select_context({'passages':passages},mode='full')
        value=engine.infer(nli_messages(context,row['claim']),method='head',head=selected_head)
        features.append((row['id'],mode,value['hidden'].copy(),value['option_logits'].copy()))
        assert not value['invalid_output']
        if cap:assert value['context_tokens']<=cap
        return {'id':row['id'],'label':value['label'],'probabilities':value['probabilities'],
            'context_tokens':value['context_tokens'],'latency_seconds':time.perf_counter()-began,
            'selected_passage_ids':[p['id'] for p in selected],'pages':[p['page'] for p in selected],
            'prompt_sha256':hashlib.sha256(json.dumps(nli_messages(context,row['claim']),ensure_ascii=False).encode()).hexdigest()}
    predictions={'neighbors':[],'diverse':[]};redundancy=[]
    def grams(p):
        tokens=words(p['text']);return {tuple(tokens[i:i+5]) for i in range(max(0,len(tokens)-4))}
    def redundant(a,b):
        if a['id']==b['id']:return True
        if a.get('page')==b.get('page') and all(k in a and k in b for k in ('char_start','char_end')):
            overlap=max(0,min(a['char_end'],b['char_end'])-max(a['char_start'],b['char_start']))
            length=min(a['char_end']-a['char_start'],b['char_end']-b['char_start'])
            if length and overlap/length>=.5:return True
        ga,gb=grams(a),grams(b);union=ga|gb
        return bool(union) and len(ga&gb)/len(union)>=.55
    for index,row in enumerate(sample):
        began=time.perf_counter();ranked=retriever.retrieve(row['passages'],row['claim'],k=30,mode='hybrid')
        lookup={p['id']:i for i,p in enumerate(row['passages'])};neighbors=[];seen=set()
        for seed in ranked[:6]:
            i=lookup[seed['id']]
            for j in (i,i-1,i+1):
                if 0<=j<len(row['passages']) and row['passages'][j]['id'] not in seen:
                    passage=row['passages'][j];neighbors.append(passage);seen.add(passage['id'])
        diverse=[];suppressed=0
        for passage in ranked:
            if any(redundant(passage,previous) for previous in diverse):suppressed+=1;continue
            diverse.append(passage)
            if len(diverse)==20:break
        preparation=time.perf_counter()-began
        for name,candidates in [('neighbors',neighbors),('diverse',diverse)]:
            value=inferred(row,candidates,head4,4096,name);value['latency_seconds']+=preparation
            predictions[name].append(value)
        redundant_pairs=sum(redundant(a,b) for i,a in enumerate(ranked[:20]) for b in ranked[:20][i+1:])
        redundancy.append({'id':row['id'],'redundant_pairs_among_top20':redundant_pairs,
            'diversity_suppressed':suppressed,'diverse_candidates':len(diverse)})
        if (index+1)%10==0:print('PHASE2_NEIGHBOR_DIVERSITY_DIAGNOSTIC',index+1,len(sample),flush=True)
    evidence_sample=[]
    for lang in ('de','fr','it'):
        for label in (0,2):
            subset=[r for r in rows if r['claim_language']==lang and candidate_by[r['id']]['label']==label]
            seen=set();unique=[]
            for row in subset:
                key=(row['document_id'],' '.join(words(row['claim'])))
                if key not in seen:seen.add(key);unique.append(row)
            evidence_sample.extend(unique[:2])
    assert len(evidence_sample)<=12
    evidence_predictions=[];evidence_proofs=[]
    evidence_cache=None
    if condition!='full':
        evidence_cache={p['id']:p['selected_passages'] for p in [json.loads(s) for s in (Path('track_2a/experiments/apertus-v15-phase2-context-v1')/condition/'validation-evidence.jsonl').read_text().splitlines()]}
    for row in evidence_sample:
        selected=evidence_cache[row['id']] if evidence_cache is not None else select_context(row,mode='full')[0]
        began=time.perf_counter();ranked=retriever.retrieve(selected,row['claim'],k=2,mode='hybrid')
        retrieval_seconds=time.perf_counter()-began
        value=inferred(row,ranked,head,CONDITIONS[condition][0] if condition!='full' else None,'evidence-only')
        value['latency_seconds']+=retrieval_seconds;evidence_predictions.append(value)
        evidence_proofs.append({'id':row['id'],'original_label':candidate_by[row['id']]['label'],
            'evidence_only_label':value['label'],'decision_preserved':value['label']==candidate_by[row['id']]['label'],
            'claim_language':row['claim_language'],'document_language':row['document_language'],
            'proposed_evidence':ranked,'output':value})
        print('PHASE2_EVIDENCE_ONLY_DIAGNOSTIC',row['id'],value['label'],flush=True)
    results={'scope':'Validation-only bounded diagnostics, not architecture selection or independent-test estimates. No training and no consumed310.',
        'sample_method':'First10 validation rows per gold-class/claim-language stratum;90 predetermined rows. Labels used only for analysis sampling, never messages.',
        'unique_sample_claim_document_pairs':len({(r['document_id'],' '.join(words(r['claim']))) for r in sample}),
        'sample_event_counts':{event:sum(r['booklet_id']==event for r in sample) for event in sorted({r['booklet_id'] for r in sample})},
        'baseline_hybrid4k_same90':evaluate(sample,[baseline_by[r['id']] for r in sample]),
        'neighbors_same90':evaluate(sample,predictions['neighbors']),
        'diversity_same90':evaluate(sample,predictions['diverse']),
        'neighbors_policy':'Six highest hybrid-ranked seeds, each seed/previous/next in source order, deduplicated, exact whole quotes capped4096; unchanged matched direct20 hybrid4k head.',
        'diversity_policy':'Top30 hybrid candidates, accept up to20 while suppressing same-page overlap>=50% of shorter quote or word5gram Jaccard>=.55; exact whole quotes capped4096.',
        'redundancy_diagnostics':redundancy,'no_new_head_fitted':True,'mechanism_adopted':False,
        'timing_caution':'Neighbor/diversity request adds shared measured retrieval/preparation plus actual encoding/head; baseline is cached. Not paired live all-inference latency.',
        'evidence_policy':'Rerank only classifier-input passages with frozen E5/hybrid and propose top2. First2 distinct normalized claim/document pairs predicted Entailment/Contradiction per claim-language; up to12 actual evidence-only forwards with same frozen head.',
        'evidence_sufficiency_n':len(evidence_proofs),
        'evidence_decision_preservation_rate':sum(p['decision_preserved'] for p in evidence_proofs)/len(evidence_proofs) if evidence_proofs else None,
        'evidence_only_original_label_metrics':evaluate(evidence_sample,evidence_predictions) if evidence_sample else None,
        'evidence_proofs':evidence_proofs,
        'evidence_caution':'Preservation is a model sensitivity diagnostic, not human proof of sufficiency or attribution. A changed decision does not by itself invalidate source evidence. No absence proof for Neutral.',
        'gpu_forward_count':2*len(sample)+len(evidence_proofs),
        'tokens_p50_p95':{name:np.quantile([p['context_tokens'] for p in pred],[.5,.95]).tolist() for name,pred in predictions.items()}}
    for name,pred in predictions.items():
        (root/f'{name}-diagnostic-predictions.jsonl').write_text(''.join(json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n' for p in pred))
    np.savez_compressed(root/'bounded-diagnostic-features.npz',ids=np.asarray([r[0] for r in features]),
        modes=np.asarray([r[1] for r in features]),hidden=np.asarray([r[2] for r in features],dtype=np.float32),
        option_logits=np.asarray([r[3] for r in features],dtype=np.float32))
    (root/'bounded-context-evidence-diagnostics.json').write_text(json.dumps(results,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return results
