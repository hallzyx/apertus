"""Resume a frozen internal holdout, preserving each completed prediction verbatim."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate_prefix(rows,predictions):
    if len(predictions)>len(rows) or [p.get('id') for p in predictions]!=[r['id'] for r in rows[:len(predictions)]]:
        raise ValueError('Saved predictions must be the unique canonical test prefix')
    for row,pred in zip(rows,predictions):
        if type(pred.get('label')) is not int or pred['label'] not in (0,1,2) or pred.get('invalid_output'):
            raise ValueError('Saved prediction is not a valid frozen decision')
        for field in ('context_tokens','latency_seconds'):
            x=pred.get(field)
            if type(x) not in (int,float) or not math.isfinite(x) or x<0:
                raise ValueError('Saved prediction lacks real runtime/token metadata')
        for field in ('evidence_ids','evidence_pages','truncated'):
            if pred.get(field)!=row.get(field):raise ValueError('Saved frozen input provenance changed')


def main():
    from ost_nli.data import load_dataset,fingerprint
    from ost_nli.metrics import evaluate
    from ost_nli.v15 import Engine,REPO,REVISION
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',default='/workspace');a=p.parse_args()
    work=Path(a.workspace);repo=Path(__file__).resolve().parents[2];os.chdir(repo)
    out=repo/'track_2a/experiments/apertus-v15-final-v1';contract=json.loads((repo/'track_2a/scripts/v15_resume_contract.json').read_text())
    selection_path=out/'selection.json';selection=json.loads(selection_path.read_text());choice=selection['choice']
    if digest(selection_path)!=contract['selection_sha256'] or selection.get('test_used_for_selection'):
        raise ValueError('The pre-test frozen selection cannot change')
    if selection['model']!=REPO or selection['model_revision']!=REVISION:
        raise ValueError('Wrong original frozen model')
    head_path=Path(choice['head']);head=json.loads(head_path.read_text())
    if digest(head_path)!=contract['head_sha256'] or head['model_revision']!=REVISION:
        raise ValueError('The selected trained head cannot change after test inference begins')
    frozen=json.loads((work/'v15-inputs/manifest.json').read_text());test_path=work/'v15-inputs'/f'test-{choice["context"]}.jsonl'
    if digest(test_path)!=frozen['splits']['test']['files'][choice['context']]['sha256']:
        raise ValueError('Original test input checksum changed')
    rows=[json.loads(x) for x in test_path.read_text().splitlines()]
    original=load_dataset(repo/'track_2a/data/private/full-booklets-v2/test.jsonl')
    if fingerprint(original)!=contract['test_dataset_sha256'] or fingerprint(original)!=frozen['splits']['test']['full_sha256']:
        raise ValueError('Canonical test fingerprint changed')
    if [r['id'] for r in rows]!=[r['id'] for r in original] or len(rows)!=310:
        raise ValueError('Original complete holdout alignment changed')
    train=load_dataset(repo/'track_2a/data/private/full-booklets-v2/train.jsonl');val=load_dataset(repo/'track_2a/data/private/full-booklets-v2/validation.jsonl')
    if {r['booklet_id'] for r in original}&{r['booklet_id'] for r in train+val}:raise ValueError('Test booklet leakage')
    predictions_path=out/'predictions.jsonl';saved=predictions_path.read_bytes();chunks=saved.splitlines(keepends=True)
    initial=contract['persisted_prefix_n']
    if len(chunks)<initial or hashlib.sha256(b''.join(chunks[:initial])).hexdigest()!=contract['persisted_prefix_sha256']:
        raise ValueError('Original completed holdout prefix cannot change')
    predictions=[json.loads(x) for x in chunks];validate_prefix(rows,predictions)
    resume_started_n=len(predictions);began=time.perf_counter()
    print(f'V15_RESUME_FROZEN_TEST preserved={resume_started_n} remaining={len(rows)-resume_started_n} head_sha256={contract["head_sha256"]}',flush=True)
    engine=Engine(work/'model')
    for i in range(resume_started_n,len(rows)):
        row=rows[i];result=engine.infer(row['messages'],choice['method'],head)
        result.pop('hidden');result.pop('option_logits')
        if result['label'] is None:raise ValueError('No fallback label for invalid final decision')
        result.update(id=row['id'],evidence_ids=row['evidence_ids'],evidence_pages=row['evidence_pages'],truncated=row['truncated'],
            model_inference_seconds=result['latency_seconds'],retrieval_seconds=row['retrieval_seconds'])
        result['latency_seconds']+=row['retrieval_seconds']
        with predictions_path.open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
        predictions.append(result)
        if (i+1)%25==0:print(f'V15_FINAL_TEST {i+1}/{len(rows)}',flush=True)
    validate_prefix(rows,predictions)
    if not predictions_path.read_bytes().startswith(saved):raise ValueError('Completed predictions were overwritten')
    metrics=evaluate(original,predictions);save(out/'metrics.json',metrics)
    audit={'logical_heldout_evaluations':1,'persisted_predictions_before_interruption':initial,'preserved_verbatim':True,
        'completed_predictions_reinferred':False,'model_or_head_refitted':False,'selection_changed':False,'test_used_for_selection':False,
        'selection_sha256':contract['selection_sha256'],'head_sha256':contract['head_sha256'],
        'initial_prefix_sha256':contract['persisted_prefix_sha256'],'source_resume_script_sha256':digest(Path(__file__)),
        'reason':'Cloud controller reboot interrupted the original frozen holdout; resume appends only its missing canonical IDs.'}
    save(out/'resume_audit.json',audit)
    save(out/'experiment.json',{'experiment_id':out.name,'status':'completed','model':REPO,'model_revision':REVISION,
        'git_commit':selection['git_commit'],'git_working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'source_resume_script_sha256':digest(Path(__file__)),'dataset_split':'strict_internal_test','n':len(rows),
        'test_sha256':fingerprint(original),'selection':choice,'macro_f1':metrics['macro_f1'],
        'resume_runtime_seconds':time.perf_counter()-began,'hardware':engine.metadata,'official_challenge_score':False,
        'context_preparation':'Original frozen real booklet PDF text; booklet and claim only',
        'latency_definition':'Actual per-row model inference plus separately measured CPU retrieval; excludes restart, startup, download and PDF parsing',
        'notes':'One logical fixed internal holdout evaluation resumed after interruption. Original persisted predictions unchanged; no test fitting or post-test architecture changes.',
        'resume_audit':audit})
    from serve_v15 import make_server
    server=make_server(engine,choice['method'],head,8001);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env={**os.environ,'LLM_NAME':REPO,'LLM_BASE_URL':'http://127.0.0.1:8001/v1','LLM_API_KEY':'',
        'EMBEDDING_MODEL_DIR':'/workspace/e5','APERTUS_REQUIRED_GENERATION':'v1.5'}
    chosen=[next(r for r in original if r['claim_language'].lower()==language) for language in ('de','fr','it')]
    batch=[]
    for row in chosen:
        pdf=repo/'track_2a/data/private/full-booklets-v2'/(hashlib.sha256(row['source_booklet_url'].encode()).hexdigest()+'.pdf')
        batch.append({'id':row['id'],'document':str(pdf),'claim':row['claim']})
    batchfile=work/'v15-real-cli.jsonl';batchfile.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in batch))
    resultfile=out/'real_cli_batch.json'
    if not resultfile.exists():
        subprocess.run(['python','-m','ost_nli','predict-batch',str(batchfile),'--output',str(resultfile),'--context',choice['context']],env=env,check=True)
    live=json.loads(resultfile.read_text())['predictions'];expected={r['id']:r for r in predictions}
    if len(live)!=3:raise ValueError('Complete real multilingual CLI output required')
    for result,row in zip(live,chosen):
        if result['id']!=row['id'] or result['label']!=expected[row['id']]['label']:
            raise ValueError('Actual PDF CLI must reproduce the frozen decision')
        for passage in result['evidence']:
            source=next(p for p in row['passages'] if p['id']==passage['id'])
            if not source['text'].startswith(passage['text']) or passage['page']!=source['page']:
                raise ValueError('Public CLI quote provenance failure')
    save(out/'live_validation.json',{'real_apertus_v15':True,'production_input':'PDF + claim only','languages':['de','fr','it'],
        'n':3,'labels_match_frozen_test':True,'source_quotes_and_pages_verified':True})
    server.shutdown();server.server_close()
    exported=repo/'track_2a/experiments/apertus-v15-research-v1'
    if exported.exists():raise ValueError('Existing final export cannot be overwritten')
    exported.mkdir()
    for name,source in [('validation',work/'v15-validation'),('training',work/'v15-train'),('feature-cache',work/'v15-feature-cache'),('final',out)]:
        shutil.copytree(source,exported/name)
    for name,source in [('model_manifest.json',work/'model/verified_manifest.json'),('inputs_manifest.json',work/'v15-inputs/manifest.json')]:
        shutil.copyfile(source,exported/name)
    for candidate in selection['head_candidates']:shutil.copytree(Path(candidate['head']).parent,exported/candidate['name'])
    files=[{'path':str(f.relative_to(exported)),'sha256':digest(f),'bytes':f.stat().st_size} for f in sorted(exported.rglob('*')) if f.is_file()]
    save(exported/'experiment.json',{'status':'completed','model':REPO,'model_revision':REVISION,'git_commit':selection['git_commit'],
        'files':files,'test_evaluation_count':1,'macro_f1':metrics['macro_f1'],'resume_audit':audit})
    subprocess.run(['python','track_2a/scripts/export_vast_cache.py','--relative-root',str(exported.relative_to(repo))],check=True)
    print(f'V15_RESEARCH_COMPLETE macro_f1={metrics["macro_f1"]:.6f}',flush=True)


if __name__=='__main__':main()
