"""One bounded, actual GPU/browser demo; no fitting or benchmark evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

RECORD='--record' in sys.argv
ROOT=Path('track_2a/experiments/apertus-live-demo-recorded-v1' if RECORD else 'track_2a/experiments/apertus-live-demo-v1')
PDF_URL='https://www.bk.admin.ch/dam/fr/sd-web/MDRXrtFaVPNG/2026-03-08_explications_du_conseil_federal.pdf'
PDF_SHA='2ec02c4d1e2cffae67f89fc5b07af117fa455f5a925cdcd3db0938e1855d053b'
HEAD_SHA='2f6a3ede1bab5e69e86938c6af8ce46a468158a40d176fe3883a593cc8dafcc1'

def run(command):
    subprocess.run(command,check=True)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    token=os.environ.pop('HF_TOKEN','')
    if not token:raise ValueError('Provider-managed HF_TOKEN missing')
    os.environ.update(PYTHONPATH='track_2a/src',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HOME='/workspace/hf',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    for lock in ('requirements-v15-cuda.lock','requirements-v15.lock','requirements-head-v15.lock','requirements-pdf.lock'):
        run(['python','-m','pip','install','--no-cache-dir','--require-hashes','-r','track_2a/'+lock])
    run(['python','track_2a/scripts/check_v15_cuda.py'])
    run(['python','-m','pip','install','--no-cache-dir','playwright==1.62.0'])
    run(['python','-m','playwright','install','--with-deps','chromium'])
    from ost_nli.v15 import download,Engine
    print('LIVE_DEMO_VERIFY_MODEL_START',flush=True)
    try:download('/workspace/model',token)
    finally:del token
    print('LIVE_DEMO_MODEL_VERIFIED',flush=True)
    from ost_nli.dense import download as download_e5,MultilingualRetriever
    download_e5('/workspace/e5')
    print('LIVE_DEMO_RETRIEVER_VERIFIED',flush=True)
    data=Path('track_2a/data/demo/votation-2026-03-08-fr.pdf').read_bytes()
    if hashlib.sha256(data).hexdigest()!=PDF_SHA:raise ValueError('Official demo PDF changed')
    pdf=Path('/workspace/demo-booklet.pdf');pdf.write_bytes(data)
    head_path=Path('track_2a/deployment/head-v15.json');head_bytes=head_path.read_bytes()
    if hashlib.sha256(head_bytes).hexdigest()!=HEAD_SHA:raise ValueError('Selected head changed')
    head=json.loads(head_bytes)
    engine=Engine('/workspace/model','cuda')
    print('LIVE_DEMO_ENGINE_READY',flush=True)
    from ost_nli.context_budget import DocumentPipeline
    from ost_nli.native_service import make_server
    pipeline=DocumentPipeline(engine,head,MultilingualRetriever('/workspace/e5'))
    backend=make_server(engine,'head',head,0,pipeline)
    backend_thread=threading.Thread(target=backend.serve_forever,daemon=True);backend_thread.start()
    os.environ['FROZEN_BASE_URL']=f'http://127.0.0.1:{backend.server_port}'
    os.environ.pop('LLM_API_KEY',None)
    from ost_nli.web import Handler
    frontend=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    frontend_thread=threading.Thread(target=frontend.serve_forever,daemon=True);frontend_thread.start()
    from playwright.sync_api import sync_playwright
    record={'status':'running','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'pdf_url':PDF_URL,'pdf_sha256':PDF_SHA,'head_sha256':HEAD_SHA,'model':engine.metadata,'context':'hybrid-8k','training_performed':False,'holdout_accessed':False,'scope':'Actual current app browser PDF upload, native Apertus inference, source quotes and JSON download. Unannotated demo claims, not accuracy evaluation.','predictions':[],'recorded_repeat':RECORD}
    (ROOT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(headless=True,args=['--no-sandbox'])
            errors=[]
            claims=[('de','Die schweizerische Währung ist der Franken.'),('fr','Le contre-projet direct propose d’inscrire dans la Constitution que la monnaie suisse est le franc.'),('it','Il controprogetto diretto stabilisce che la valuta svizzera è l’euro.')]
            for pass_number in ((1,2) if RECORD else (1,)):
                capture=RECORD and pass_number==2
                prefix=f'pass-{pass_number}'
                page=browser.new_page(locale='es-PE',viewport={'width':1440,'height':1100},**({'record_video_dir':str(ROOT/'video'),'record_video_size':{'width':1440,'height':1100}} if capture else {}))
                page.on('pageerror',lambda error:errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{frontend.server_port}',wait_until='networkidle')
                if capture:page.wait_for_timeout(2500)
                page.screenshot(path=str(ROOT/(prefix+'-01-start.png')),full_page=True)
                page.locator('#pdf').scroll_into_view_if_needed()
                page.locator('#pdf').set_input_files(str(pdf))
                page.wait_for_function("document.getElementById('doc').value.length>0 && !document.getElementById('claim').disabled",timeout=90000)
                document=json.loads(page.locator('#doc').input_value());lookup={p['id']:p for p in document['passages']}
                record['pdf_pages']=document['pages'];record['document_passages']=len(document['passages'])
                if capture:page.wait_for_timeout(2500)
                for index,(language,claim) in enumerate(claims,1):
                    page.locator('#claim').scroll_into_view_if_needed()
                    page.locator('#claim').fill(claim)
                    if capture:page.wait_for_timeout(2500)
                    if index==1:page.screenshot(path=str(ROOT/(prefix+'-02-pdf-and-claim.png')),full_page=True)
                    if page.locator('#predict').is_disabled():raise ValueError('Actual native backend not configured in browser')
                    page.locator('#predict').click()
                    page.wait_for_function("!document.getElementById('raw').hidden",timeout=180000)
                    result=json.loads(page.locator('#result').text_content())
                    assert result['model']=='swiss-ai/Apertus-v1.5-8B' and result['label'] in (0,1,2)
                    assert result['input_tokens']>0 and result['inference_time_ms']>0
                    for evidence in result['evidence']:
                        source=lookup[evidence['id']]
                        assert evidence['text']==source['text'] and evidence['page']==source['page']
                    with page.expect_download() as download:page.locator('#download').click()
                    output=ROOT/f'{prefix}-prediction-{language}.json';download.value.save_as(str(output))
                    assert json.loads(output.read_text())==result
                    page.evaluate("document.getElementById('results').scrollIntoView({block:'start'})")
                    page.screenshot(path=str(ROOT/f'{prefix}-03-result-{language}.png'),full_page=False)
                    if capture:
                        page.wait_for_timeout(4000)
                        page.locator('.passage').first.scroll_into_view_if_needed()
                        page.wait_for_timeout(2500)
                    record['predictions'].append({'pass':pass_number,'claim_language':language,'claim':claim,'label':result['label'],'input_tokens':result['input_tokens'],'inference_time_ms':result['inference_time_ms'],'exact_source_quotes_pages_verified':True,'download_json_equal':True,'output':output.name})
                    (ROOT/'verification.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
                    print('LIVE_DEMO_REAL_BROWSER_PREDICTION','pass',pass_number,language,'label',result['label'],'tokens',result['input_tokens'],flush=True)
                if RECORD and pass_number==1:
                    print('LIVE_DEMO_FIRST_PASS_VERIFIED_BACKEND_REMAINS_RUNNING',flush=True)
                video=page.video
                page.close()
                if capture:
                    video.save_as(str(ROOT/'demo.webm'))
                    shutil.rmtree(ROOT/'video')
                    record['video']='demo.webm'
            if RECORD:
                first=[p['label'] for p in record['predictions'] if p['pass']==1]
                second=[p['label'] for p in record['predictions'] if p['pass']==2]
                assert first==second
                record['second_pass_labels_reproduced']=True
                record['same_native_backend_kept_running_between_passes']=True
            assert not errors
            record.update(status='completed',browser_errors=errors,browser_version=browser.version,actual_native_inference=True)
            browser.close()
    finally:
        frontend.shutdown();frontend.server_close();backend.shutdown();backend.server_close()
    record['files']=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='verification.json']
    (ROOT/'verification.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
    print('LIVE_DEMO_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    finally:
        if ROOT.exists():run(['python','track_2a/scripts/export_vast_cache.py','--relative-root',str(ROOT)])
