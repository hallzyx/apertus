"""Deploy only the already-validated registered phase2 decision, without training."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path('track_2a');phase=root/'experiments/apertus-v15-phase2'
    result_path=phase/'final-results.json'
    accuracy_path=phase/'accuracy-results.json'
    if accuracy_path.exists():
        alternative=json.loads(accuracy_path.read_text())
        if alternative['final_choice']=='hybrid-8k' and not alternative['guard_failures']:result_path=accuracy_path
    final=json.loads(result_path.read_text())
    proposal=json.loads((phase/'context-results.json').read_text())
    assert final['status']=='completed_frozen_final_validation' and not final['consumed310_accessed']
    selection_path=root/'deployment/v15-selection.json';old=json.loads(selection_path.read_text())
    if old.get('phase2_completed'):raise ValueError('Do not change an already frozen phase2 selection')
    condition=final['final_choice']
    path=Path(proposal['comparisons'][condition]['source_head']) if condition!='full' else root/'experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/head.json'
    blob=path.read_bytes();head=json.loads(blob);assert head['context']==condition
    if condition!='full':
        assert not final['guard_failures']
        assert hashlib.sha256(blob).hexdigest()==final['candidate_head_sha256']
    history={k:old.get(k) for k in ('choice','internal_test_macro_f1','internal_holdout_complete','source_head_sha256','status')}
    (root/'deployment/head-v15.json').write_bytes(blob)
    metrics=final['final_validation_metrics']
    matched_base=final['selected_context_base_metrics']['macro_f1']
    old.update(phase2_validation_macro_f1=metrics['macro_f1'],phase2_matched_base_macro_f1=matched_base,
        phase2_head_gain_over_same_context_base=metrics['macro_f1']-matched_base,
        phase2_gain_over_original_full_head=metrics['macro_f1']-.8944609902440983,
        phase2_gain_scope='Development validation only; no matched compact held-out experiment')
    name=old['choice']['name'] if condition=='full' else f'phase2-{condition}-matched-head'
    choice={'name':name,'context':condition,
        'method':'head','head':'track_2a/deployment/head-v15.json','feature_kind':'hidden',
        'macro_f1':metrics['macro_f1'],'tokens':metrics['average_context_tokens']}
    if condition!='full':choice['transport']='document-service'
    old.update(choice=choice,phase2_completed=True,status='phase2_frozen_after_registered_validation',
        phase2_results=str(result_path.relative_to(root)),
        phase2_result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
        source_head_sha256=hashlib.sha256(blob).hexdigest(),historical_original_full=history,
        original_consumed310_score_applies_to_current_head=condition=='full',
        phase2_consumed310_accessed=False,lora='Not performed; all Apertus weights frozen')
    if condition!='full':
        old['historical_source_cache']=old.get('source_cache')
        old['source_cache']=json.loads((path.parent.parent/condition/'experiment.json').read_text())
        old.update(internal_test_macro_f1=None,internal_holdout_complete=False,
            head_macro_f1_gain=None,head_candidates_scope='Historical phase1 candidates; phase2 candidates recorded in context-results.json')
    selection_path.write_text(json.dumps(old,indent=2,allow_nan=False)+'\n')
    benchmark_path=root/'deployment/benchmark.json';benchmark=json.loads(benchmark_path.read_text())
    benchmark.update(selected_experiment_id=choice['name'],selected_context=condition,
        scope='Internal validation architecture development; historical310 score applies only to original full head, not a newly selected compact head. Not organizer hidden evaluation.',
        phase2_complete=True,phase2_final_validation_macro_f1=metrics['macro_f1'])
    for name,entry in proposal['comparisons'].items():
        m=entry['matched_head'];benchmark['results'].append({'experiment':f'phase2-{name}-matched-head',
            'split':'validation','n':276,'macro_f1':m['macro_f1'],
            'mean_context_tokens':m['average_context_tokens'],'mean_inference_seconds':m['average_latency_seconds'],
            'scope':'A6000 cached encoding + retrieval + CPU head; not live full request timing'})
    benchmark_path.write_text(json.dumps(benchmark,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'frozen_choice':condition,'head_sha256':old['source_head_sha256'],'consumed310_accessed':False}))


if __name__=='__main__':main()
