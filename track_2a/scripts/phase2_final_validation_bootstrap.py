"""Frozen provisional candidate: matched wrong-booklet controls and real PDF CLI."""
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from phase2_remote_bootstrap import run


def main():
    proposal=json.loads(Path('track_2a/experiments/apertus-v15-phase2/context-results.json').read_text())
    condition=proposal['provisional_choice']
    source_head=Path(proposal['comparisons'][condition]['source_head']) if condition!='full' else Path('track_2a/deployment/head-v15.json')
    head_bytes=source_head.read_bytes()
    expected_sha=proposal['comparisons'][condition]['head_sha256'] if condition!='full' else json.loads(Path('track_2a/experiments/apertus-v15-phase2-controls-v1/contract.json').read_text())['head_sha256']
    if hashlib.sha256(head_bytes).hexdigest()!=expected_sha:
        raise ValueError('Frozen candidate head changed')
    head=json.loads(head_bytes)
    token=os.environ.pop('HF_TOKEN','')
    if not token:raise ValueError('Provider-managed model read credential required')
    os.environ.update(PYTHONPATH='track_2a/src',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HOME='/workspace/hf',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for lock in ('requirements-v15-cuda.lock','requirements-v15.lock','requirements-head-v15.lock','requirements-pdf.lock'):
        run(['python','-m','pip','install','--no-cache-dir','--require-hashes','-r','track_2a/'+lock])
    run(['python','track_2a/scripts/check_v15_cuda.py'])
    from ost_nli.v15 import download,Engine
    try:download('/workspace/model',token)
    finally:del token
    from ost_nli.dense import download as download_e5,MultilingualRetriever
    retriever=None
    if condition.startswith('hybrid-'):
        download_e5('/workspace/e5');retriever=MultilingualRetriever('/workspace/e5')
    source=Path('/workspace/phase2-source');full=Path('/workspace/phase2-booklets')
    run(['python','track_2a/scripts/prepare_phase2_source.py','--output',str(source)])
    run(['python','track_2a/scripts/prepare_booklets.py','--source',str(source/'selected-official.jsonl'),
         '--splits',str(source),'--output',str(full),'--partitions','validation'])
    from ost_nli.data import load_dataset,fingerprint
    from ost_nli.metrics import evaluate
    rows=load_dataset(full/'validation.jsonl')
    if fingerprint(rows)!='00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f':
        raise ValueError('Original validation changed')
    import numpy as np
    engine=Engine('/workspace/model','cuda')
    from ost_nli.context_budget import DocumentPipeline
    pipeline=DocumentPipeline(engine,head,retriever)
    root=Path('track_2a/experiments/apertus-v15-phase2-final-validation-v1');root.mkdir(parents=True)
    documents={r['document_id']:r for r in rows}
    record={'status':'started','context':condition,'head_sha256':hashlib.sha256(head_bytes).hexdigest(),
        'validation_sha256':fingerprint(rows),'model_metadata':engine.metadata,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'engine_sha256':hashlib.sha256(Path('track_2a/src/ost_nli/v15.py').read_bytes()).hexdigest(),
        'pipeline_sha256':hashlib.sha256(Path('track_2a/src/ost_nli/context_budget.py').read_bytes()).hexdigest(),
        'training_performed':False,'holdout_accessed':False,
        'interpretation':'Original labels measure retention under swapped premises; actual swapped-pair NLI labels unknown'}
    (root/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
    original_infer=engine.infer;last={}
    def captured_infer(*args,**kwargs):
        result=original_infer(*args,**kwargs)
        last['hidden']=result['hidden'].copy();last['option_logits']=result['option_logits'].copy()
        return result
    engine.infer=captured_infer
    original_head=json.loads(Path('track_2a/deployment/head-v15.json').read_text())
    assert hashlib.sha256(Path('track_2a/deployment/head-v15.json').read_bytes()).hexdigest()==json.loads(Path('track_2a/experiments/apertus-v15-phase2-controls-v1/contract.json').read_text())['head_sha256']
    full_pipeline=DocumentPipeline(engine,original_head)
    correct=[];correct_hidden=[];correct_logits=[]
    for index,row in enumerate(rows):
        result=full_pipeline.predict(row,row['claim']);result['id']=row['id']
        correct_hidden.append(last['hidden']);correct_logits.append(last['option_logits'])
        scores=last['option_logits'].astype(np.float64);q=np.exp(scores-scores.max());q/=q.sum()
        result['native_base_scores_label']=int(q.argmax());result['native_base_scores_probabilities']=q.tolist();correct.append(result)
        with (root/'correct-full-recheck.jsonl').open('a') as file:file.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
        if (index+1)%20==0 or index+1==len(rows):print('PHASE2_FULL_RECHECK',index+1,len(rows),flush=True)
    (root/'correct-full-recheck-metrics.json').write_text(json.dumps(evaluate(rows,correct),indent=2)+'\n')
    np.savez_compressed(root/'correct-full-recheck.npz',ids=np.asarray([r['id'] for r in rows]),
        labels=np.asarray([r['label'] for r in rows]),hidden=np.asarray(correct_hidden,dtype=np.float32),
        option_logits=np.asarray(correct_logits,dtype=np.float32))
    for seed in ([42,1337] if condition!='full' else []):
        folder=root/f'wrong-{seed}';folder.mkdir();predictions=[];hidden=[];logits=[]
        for index,row in enumerate(rows):
            candidates=sorted((r for r in documents.values() if r['booklet_id']!=row['booklet_id']
                and r['document_language']==row['document_language']),key=lambda r:r['document_id'])
            donor=random.Random(f'{seed}:{row["id"]}').choice(candidates)
            result=pipeline.predict(donor,row['claim'])
            result.update(id=row['id'],donor_document_id=donor['document_id'],donor_booklet_id=donor['booklet_id'],
                donor_language=donor['document_language'],gold_document_id=row['document_id'])
            predictions.append(result);hidden.append(last['hidden']);logits.append(last['option_logits'])
            with (folder/'predictions.jsonl').open('a') as file:file.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
            if (index+1)%20==0 or index+1==len(rows):print('PHASE2_FINAL_CONTROL',seed,index+1,len(rows),flush=True)
        np.savez_compressed(folder/'validation.npz',ids=np.asarray([r['id'] for r in rows]),
            labels=np.asarray([r['label'] for r in rows]),hidden=np.asarray(hidden,dtype=np.float32),
            option_logits=np.asarray(logits,dtype=np.float32))
        (folder/'metrics.json').write_text(json.dumps(evaluate(rows,predictions),indent=2)+'\n')
    from ost_nli.native_service import make_server
    server=make_server(engine,'head',head,0,pipeline)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    integration=[]
    try:
        expected_path=source_head.parent/'predictions.json' if condition!='full' else Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/predictions.json')
        expected={r['id']:r for r in json.loads(expected_path.read_text())['predictions']}
        for claim_lang,doc_lang in [('de','fr'),('fr','it'),('it','de')]:
            row=next(r for r in rows if r['claim_language']==claim_lang and r['document_language']==doc_lang)
            pdf=full/(hashlib.sha256(row['source_booklet_url'].encode()).hexdigest()+'.pdf')
            if hashlib.sha256(pdf.read_bytes()).hexdigest()!=row['source_pdf_sha256']:raise ValueError('CLI source PDF changed')
            environment={**os.environ,'FROZEN_BASE_URL':f'http://127.0.0.1:{server.server_port}',
                         'APERTUS_REQUIRED_GENERATION':'v1.5'}
            command=[sys.executable,'-m','ost_nli','predict',str(pdf),row['claim']]
            response=subprocess.run(command,env=environment,capture_output=True,text=True,timeout=180,check=True)
            value=json.loads(response.stdout)
            delta=max(abs(a-b) for a,b in zip(value['probabilities'],expected[row['id']]['probabilities']))
            proof={'id':row['id'],'claim_language':claim_lang,'document_language':doc_lang,
                'pdf_sha256':row['source_pdf_sha256'],'expected_cached_label':expected[row['id']]['label'],
                'label_reproduced':value['label']==expected[row['id']]['label'],
                'max_probability_difference':delta,'output':value}
            integration.append(proof)
            print('PHASE2_REAL_PDF_CLI',claim_lang,doc_lang,'label',value['label'],'cache_label_reproduced',proof['label_reproduced'],flush=True)
        # Exercise the actual frontend upload and prediction route, not a fake model.
        from ost_nli.web import Handler
        os.environ['FROZEN_BASE_URL']=f'http://127.0.0.1:{server.server_port}'
        front=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        front_thread=threading.Thread(target=front.serve_forever,daemon=True);front_thread.start()
        try:
            row=next(r for r in rows if r['claim_language']=='de' and r['document_language']=='fr')
            pdf=full/(hashlib.sha256(row['source_booklet_url'].encode()).hexdigest()+'.pdf')
            base=f'http://127.0.0.1:{front.server_port}'
            request=urllib.request.Request(base+'/api/document',data=pdf.read_bytes(),headers={'Content-Type':'application/pdf'})
            document=json.load(urllib.request.urlopen(request,timeout=180))
            request=urllib.request.Request(base+'/api/predict',data=json.dumps({'document':document,'claim':row['claim']}).encode(),headers={'Content-Type':'application/json'})
            value=json.load(urllib.request.urlopen(request,timeout=180))
            proof={'scope':'Actual frontend PDF upload + native Apertus inference; browser visual rendering separate',
                'id':row['id'],'pdf_sha256':row['source_pdf_sha256'],'pages':document['pages'],
                'label_reproduced':value['label']==expected[row['id']]['label'],'output':value}
            (root/'frontend-real-inference.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
            print('PHASE2_REAL_FRONTEND_INFERENCE',value['label'],flush=True)
        finally:front.shutdown();front.server_close();front_thread.join()
    finally:server.shutdown();server.server_close();thread.join()
    (root/'cli-integration.json').write_text(json.dumps(integration,indent=2,allow_nan=False)+'\n')
    (root/'model_manifest.json').write_bytes(Path('/workspace/model/verified_manifest.json').read_bytes())
    if retriever:(root/'retriever_manifest.json').write_bytes(Path('/workspace/e5/verified_manifest.json').read_bytes())
    files=[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='experiment.json']
    record.update(status='completed',files=files,gpu_forward_count=832 if condition!='full' else 280,
        full_recheck_reason='Resolve original A40-correct versus A6000-control precision/hardware confound; validation only')
    (root/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')
    run(['python','track_2a/scripts/export_vast_cache.py','--relative-root',str(root)])
    print('PHASE2_FINAL_VALIDATION_RESULTS_EXPORTED',flush=True)


if __name__=='__main__':main()
