"""Choose on validation, fit CPU heads on train, evaluate test once, exercise CLI."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    from ost_nli.data import load_dataset, fingerprint
    from ost_nli.metrics import evaluate
    from ost_nli.v15 import Engine, REPO, REVISION
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',default='/workspace')
    a=p.parse_args();work=Path(a.workspace);repo=Path(__file__).resolve().parents[2]
    os.chdir(repo);out=repo/'track_2a/experiments/apertus-v15-final-v1'
    if (out/'selection.json').exists() and (repo/'track_2a/scripts/v15_resume_contract.json').exists():
        subprocess.run(['python','track_2a/scripts/resume_v15_final.py','--workspace',str(work)],check=True)
        return
    if out.exists():raise ValueError('Final evaluation cannot be repeated or overwritten')
    out.mkdir(parents=True)
    validation=work/'v15-validation';training=work/'v15-train'
    context=json.loads((work/'v15-context-choice.json').read_text())['context']
    inputs=json.loads((work/'v15-inputs/manifest.json').read_text())
    cache=work/'v15-feature-cache';cache.mkdir()
    shutil.copyfile(training/f'{context}-score/features.npz',cache/'train.npz')
    shutil.copyfile(validation/f'{context}-score/features.npz',cache/'validation.npz')
    shutil.copyfile(validation/f'{context}-score/predictions.jsonl',cache/'validation_rows.jsonl')
    source=json.loads((training/'experiment.json').read_text())
    metadata={**source,'status':'completed','model':REPO,'model_revision':REVISION,'context':context,
        'train_sha256':inputs['splits']['train']['full_sha256'],
        'validation_sha256':inputs['splits']['validation']['full_sha256'],
        'files':[{'path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(cache.iterdir())]}
    save(cache/'experiment.json',metadata)
    subprocess.run(['python','-m','pip','install','--no-cache-dir','--require-hashes',
        '-r','track_2a/requirements-head-v15.lock'],check=True)
    heads=[]
    for features in ['option_logits','hidden']:
        headout=repo/f'track_2a/experiments/apertus-v15-{context}-{features}-v1'
        subprocess.run(['python','track_2a/scripts/train_v15_head.py','--cache',str(cache),'--context',context,
            '--features',features,'--train','track_2a/data/private/full-booklets-v2/train.jsonl',
            '--validation','track_2a/data/private/full-booklets-v2/validation.jsonl',
            '--output-dir',str(headout)],check=True)
        metrics=json.loads((headout/'metrics.json').read_text())
        heads.append({'name':headout.name,'context':context,'method':'head','head':str(headout/'head.json'),
            'feature_kind':features,'macro_f1':metrics['macro_f1'],'tokens':metrics['average_context_tokens']})
    baselines=[]
    for directory in sorted(validation.iterdir()):
        if not directory.is_dir():continue
        ctx,method=directory.name.rsplit('-',1)
        if ctx=='reference':continue
        metrics=json.loads((directory/'metrics.json').read_text())
        if metrics['invalid_outputs']:continue
        baselines.append({'name':directory.name,'context':ctx,'method':method,
            'macro_f1':metrics['macro_f1'],'tokens':metrics['average_context_tokens']})
    baseline=max(baselines,key=lambda r:(r['macro_f1'],-r['tokens'],r['method']=='score'))
    best_head=max(heads,key=lambda r:(r['macro_f1'],r['feature_kind']=='option_logits'))
    improvement=best_head['macro_f1']-baseline['macro_f1']
    choice=best_head if improvement>=.02 else baseline
    selection={'choice':choice,'best_baseline':baseline,'head_candidates':heads,
        'head_macro_f1_gain':improvement,'decision_head_gate':'GO' if improvement>=.02 else 'NO-GO',
        'gate_rule':'Use a trained head only if validation Macro-F1 improves over the best booklet-only baseline by at least 0.02; rule fixed before seeing results.',
        'test_used_for_selection':False,'lora':'Not performed; all Apertus weights remain frozen',
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'model':REPO,'model_revision':REVISION,'source_cache':metadata}
    save(out/'selection.json',selection)
    print('V15_CONFIGURATION_FROZEN '+json.dumps(choice),flush=True)
    frozen=json.loads((work/'v15-inputs/manifest.json').read_text())
    test_path=work/'v15-inputs'/f'test-{choice["context"]}.jsonl'
    if hashlib.sha256(test_path.read_bytes()).hexdigest()!=frozen['splits']['test']['files'][choice['context']]['sha256']:
        raise ValueError('Test input checksum mismatch')
    rows=[json.loads(l) for l in test_path.read_text().splitlines()]
    original=load_dataset(repo/'track_2a/data/private/full-booklets-v2/test.jsonl')
    if [r['id'] for r in rows]!=[r['id'] for r in original]:raise ValueError('Test alignment failure')
    if fingerprint(original)!=frozen['splits']['test']['full_sha256']:raise ValueError('Test dataset fingerprint mismatch')
    train=load_dataset(repo/'track_2a/data/private/full-booklets-v2/train.jsonl')
    val=load_dataset(repo/'track_2a/data/private/full-booklets-v2/validation.jsonl')
    if {r['booklet_id'] for r in original}&{r['booklet_id'] for r in train+val}:raise ValueError('Test booklet leakage')
    head=json.loads(Path(choice['head']).read_text()) if choice['method']=='head' else None
    engine=Engine(work/'model');predictions=[];began=time.perf_counter()
    for i,row in enumerate(rows):
        result=engine.infer(row['messages'],choice['method'],head)
        result.pop('hidden');result.pop('option_logits')
        if result['label'] is None:raise ValueError('Final production decision invalid; no fallback label')
        result.update(id=row['id'],evidence_ids=row['evidence_ids'],evidence_pages=row['evidence_pages'],
            truncated=row['truncated'],model_inference_seconds=result['latency_seconds'],retrieval_seconds=row['retrieval_seconds'])
        result['latency_seconds']+=row['retrieval_seconds'];predictions.append(result)
        with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
        if (i+1)%25==0:print(f'V15_FINAL_TEST {i+1}/{len(rows)}',flush=True)
    metrics=evaluate(original,predictions);save(out/'metrics.json',metrics)
    save(out/'experiment.json',{'experiment_id':out.name,'status':'completed','model':REPO,'model_revision':REVISION,
        'git_commit':selection['git_commit'],'dataset_split':'strict_internal_test','n':len(rows),
        'test_sha256':fingerprint(original),'selection':choice,'macro_f1':metrics['macro_f1'],
        'runtime_seconds':time.perf_counter()-began,'hardware':engine.metadata,'official_challenge_score':False,
        'context_preparation':'Real booklet PDF text; selected context depends only on booklet and claim',
        'latency_definition':'Model inference plus separately measured CPU retrieval; excludes startup, download and PDF parsing',
        'notes':'Single fixed internal holdout evaluation. No test fitting or post-test architecture changes.'})
    # The actual public CLI uses PDF + claim, with no reference or label fields.
    from serve_v15 import make_server
    server=make_server(engine,choice['method'],head,8001)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env={**os.environ,'LLM_NAME':REPO,'LLM_BASE_URL':'http://127.0.0.1:8001/v1',
        'LLM_API_KEY':'','EMBEDDING_MODEL_DIR':'/workspace/e5','APERTUS_REQUIRED_GENERATION':'v1.5'}
    chosen=[]
    for language in ['de','fr','it']:
        chosen.append(next(r for r in original if r['claim_language'].lower()==language))
    batch=[]
    for row in chosen:
        name=hashlib.sha256(row['source_booklet_url'].encode()).hexdigest()+'.pdf'
        pdf=next((repo/'track_2a/data/private'/directory/name for directory in
                  ('full-booklets-v2','full-booklets-v1') if
                  (repo/'track_2a/data/private'/directory/name).is_file()),None)
        if pdf is None or hashlib.sha256(pdf.read_bytes()).hexdigest()!=row['source_pdf_sha256']:
            raise ValueError('Original checksum-matched booklet PDF required for live CLI')
        batch.append({'id':row['id'],'document':str(pdf),'claim':row['claim']})
    batchfile=work/'v15-real-cli.jsonl'
    batchfile.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in batch))
    resultfile=out/'real_cli_batch.json'
    subprocess.run(['python','-m','ost_nli','predict-batch',str(batchfile),'--output',str(resultfile),
        '--context',choice['context']],env=env,check=True)
    live=json.loads(resultfile.read_text())['predictions'];expected={r['id']:r for r in predictions}
    for result,row in zip(live,chosen):
        if result['id']!=row['id'] or result['label']!=expected[row['id']]['label']:
            raise ValueError('Public PDF CLI does not reproduce the frozen prediction')
        for passage in result['evidence']:
            source=next(p for p in row['passages'] if p['id']==passage['id'])
            if not source['text'].startswith(passage['text']) or passage['page']!=source['page']:
                raise ValueError('Public CLI quote provenance failure')
    save(out/'live_validation.json',{'real_apertus_v15':True,'production_input':'PDF + claim only',
        'languages':['de','fr','it'],'n':len(live),'labels_match_frozen_test':True,'source_quotes_and_pages_verified':True})
    server.shutdown();server.server_close()
    # Keep source-checksummed artifacts ready for HTTPS stopped-instance recovery.
    exported=repo/'track_2a/experiments/apertus-v15-research-v1';exported.mkdir()
    shutil.copytree(validation,exported/'validation')
    shutil.copytree(training,exported/'training')
    shutil.copytree(cache,exported/'feature-cache')
    shutil.copyfile(work/'model/verified_manifest.json',exported/'model_manifest.json')
    shutil.copyfile(work/'v15-inputs/manifest.json',exported/'inputs_manifest.json')
    shutil.copytree(out,exported/'final')
    for candidate in heads:shutil.copytree(Path(candidate['head']).parent,exported/candidate['name'])
    files=[{'path':str(f.relative_to(exported)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in sorted(exported.rglob('*')) if f.is_file()]
    save(exported/'experiment.json',{'status':'completed','model':REPO,'model_revision':REVISION,
        'git_commit':selection['git_commit'],'files':files,'test_evaluation_count':1,'macro_f1':metrics['macro_f1']})
    subprocess.run(['python','track_2a/scripts/export_vast_cache.py','--relative-root',str(exported.relative_to(repo))],check=True)
    print(f'V15_RESEARCH_COMPLETE macro_f1={metrics["macro_f1"]:.6f}',flush=True)


if __name__=='__main__':main()
