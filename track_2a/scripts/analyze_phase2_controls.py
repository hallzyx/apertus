"""Analyze completed real controls without reopening the consumed holdout."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    from ost_nli.data import load_dataset, fingerprint
    from ost_nli.metrics import evaluate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--controls', default='track_2a/experiments/apertus-v15-phase2-controls-v1')
    p.add_argument('--validation', default='track_2a/data/private/full-booklets-v2/validation.jsonl')
    p.add_argument('--output', default='track_2a/experiments/apertus-v15-phase2/grounding-results.json')
    a = p.parse_args()
    root = Path(a.controls)
    rows = load_dataset(a.validation)
    if fingerprint(rows) != '00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f':
        raise ValueError('Frozen validation changed')
    contract = json.loads((root/'contract.json').read_text())
    if hashlib.sha256((root/'inputs_manifest.json').read_bytes()).hexdigest() != contract['input_manifest_sha256']:
        raise ValueError('Input manifest recovery mismatch')
    head_path = Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/head.json')
    if hashlib.sha256(head_path.read_bytes()).hexdigest() != contract['head_sha256']:
        raise ValueError('Historical frozen head changed')
    old_path = Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/predictions.json')
    correct = json.loads(old_path.read_text())['predictions']
    original = {r['id']:r for r in correct}
    correct_metrics = evaluate(rows,correct)
    baseline_f1 = correct_metrics['macro_f1']
    conditions = {}
    for condition in ('claim-only','wrong-42','wrong-1337','generic'):
        folder = root/condition
        manifest = json.loads((folder/'experiment.json').read_text())
        if manifest['status'] != 'completed' or manifest['validation_sha256'] != fingerprint(rows):
            raise ValueError('Incomplete/mismatched control cache')
        for e in manifest['files']:
            if hashlib.sha256((folder/e['path']).read_bytes()).hexdigest() != e['sha256']:
                raise ValueError('Control file recovery checksum mismatch')
        pred = [json.loads(s) for s in (folder/'validation_rows.jsonl').read_text().splitlines()]
        metrics = evaluate(rows,pred)
        with np.load(folder/'validation.npz',allow_pickle=False) as cache:
            assert cache['ids'].tolist() == [r['id'] for r in rows]
            assert cache['labels'].tolist() == [r['label'] for r in rows]
            scores = cache['option_logits'].astype(np.float64)
        base_probs = np.exp(scores-scores.max(1,keepdims=True))
        base_probs /= base_probs.sum(1,keepdims=True)
        base_pred = [{**r,'label':int(q.argmax()),'probabilities':q.tolist()} for r,q in zip(pred,base_probs)]
        by_id = {r['id']:r for r in pred}
        transition = [[0]*3 for _ in range(3)]
        for r in rows:
            transition[original[r['id']]['label']][by_id[r['id']]['label']] += 1
        neut = [p for p in pred if p['label']==1]
        entry = {'frozen_full_head':metrics,'restricted_base_class_scores':evaluate(rows,base_pred),
                 'absolute_original_label_f1_drop':baseline_f1-metrics['macro_f1'],
                 'relative_original_label_f1_drop':(baseline_f1-metrics['macro_f1'])/baseline_f1,
                 'original_label_f1_retention':metrics['macro_f1']/baseline_f1,
                 'prediction_change_rate':sum(original[r['id']]['label']!=by_id[r['id']]['label'] for r in rows)/len(rows),
                 'prediction_transition_correct_to_control':transition,
                 'neutral_prediction_rate':len(neut)/len(pred),
                 'mean_neutral_confidence':float(np.mean([p['probabilities'][1] for p in neut])) if neut else None,
                 'tokens_p50_p95':np.quantile([p['context_tokens'] for p in pred],[.5,.95]).tolist(),
                 'latency_p50_p95':np.quantile([p['latency_seconds'] for p in pred],[.5,.95]).tolist(),
                 'by_voting_event':{},
                 'interpretation':'Original labels measure artifact retention/sensitivity only. Swapped/empty evidence changes NLI premises; these labels are not control-condition NLI gold.'}
        for event in sorted({r['booklet_id'] for r in rows}):
            selected = [r for r in rows if r['booklet_id']==event]
            entry['by_voting_event'][event] = evaluate(selected,[by_id[r['id']] for r in selected])
        conditions[condition] = entry
    fitted_path = root/'claim-only-fitted-head/predictions.json'
    fitted = json.loads(fitted_path.read_text())
    if fitted['dataset_sha256'] != fingerprint(rows):
        raise ValueError('Claim-only head evaluated on changed split')
    claim_metrics = evaluate(rows,fitted['predictions'])
    train_audit = json.loads((root/'claim-only-fitted-head/training_audit.json').read_text())
    assert train_audit['temperature_fit_uses_validation'] is False
    compare = {'correct_document_frozen_head':baseline_f1,
               'claim_only_train_fitted_head':claim_metrics['macro_f1'],
               **{c:conditions[c]['frozen_full_head']['macro_f1'] for c in conditions}}
    warnings = [c for c,f1 in compare.items() if c not in ('correct_document_frozen_head','generic')
                and (f1 >= baseline_f1-.05 or f1/baseline_f1 >= .90)]
    wrong_scores = [conditions[c]['frozen_full_head']['macro_f1'] for c in ('wrong-42','wrong-1337')]
    result = {'status':'completed_actual_controls','scope':'validation only; original310 remains historical, unexamined',
              'comparison':compare,'correct_document':correct_metrics,'conditions':conditions,
              'claim_only_fitted_head':{'metrics':claim_metrics,'training_configuration':fitted['configuration'],
                    'absolute_f1_gap':baseline_f1-claim_metrics['macro_f1'],'retention':claim_metrics['macro_f1']/baseline_f1,
                    'interpretation':'Train-only refit tests claim-label signal capability; it is not the original frozen head under intervention'},
              'wrong_document_two_seed_macro_f1_std_population':float(np.std(wrong_scores)),
              'warning_conditions':warnings,'grounding_gate':'WARNING_PAUSE_ARCHITECTURE_OPTIMIZATION' if warnings else 'GO_WITH_LIMITATIONS',
              'gate_rule':'Registered pre-GPU: retention>=.90 or F1 within.05 triggers artifact investigation',
              'latency_caution':'Original correct-context baseline encoded on A40; new interventions on A6000. No paired causal latency improvement claim across GPUs.',
              'claim_only_cross_lingual_note':'Language-pair subsets describe inherited benchmark groups; absent document means this is not a cross-lingual grounding task',
              'model_revision':contract['engine_sha256'],'provenance':contract,
              'fine_tuning_performed':False,'holdout_used_for_development':False}
    # This field identifies model weights, not engine source code.
    result['model_revision'] = 'a411d838600baf0e3635a3daf66fb7c55fc97bb6'
    output = Path(a.output)
    if output.exists():
        raise ValueError('Do not overwrite recorded control conclusion')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'comparison':compare,'gate':result['grounding_gate'],'warnings':warnings},indent=2))


if __name__ == '__main__':
    main()
