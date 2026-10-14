"""One bounded, actual GPU/browser demo; no fitting or benchmark evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer

ROOT=Path('track_2a/experiments/apertus-live-demo-v1')
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
    try:download('/workspace/model',token)
    finally:del token
    from ost_nli.dense import download as download_e5,MultilingualRetriever
    download_e5('/workspace/e5')
    with urllib.request.urlopen(PDF_URL,timeout=90) as response:data=response.read()
    if hashlib.sha256(data).hexdigest()!=PDF_SHA:raise ValueError('Official demo PDF changed')
    pdf=Path('/workspace/demo-booklet.pdf');pdf.write_bytes(data)
    head_path=Path('track_2a/deployment/head-v15.json');head_bytes=head_path.read_bytes()
    if hashlib.sha256(head_bytes).hexdigest()!=HEAD_SHA:raise ValueError('Selected head changed')
    head=json.loads(head_bytes)
    engine=Engine('/workspace/model','cuda')
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
    record={'status':'running','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'pdf_url':PDF_URL,'pdf_sha256':PDF_SHA,'head_sha256':HEAD_SHA,'model':engine.metadata,'context':'hybrid-8k','training_performed':False,'holdout_accessed':False,'scope':'Actual current app browser PDF upload, native Apertus inference, source quotes and JSON download. Unannotated demo claims, not accuracy evaluation.','predictions':[]}
    (ROOT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        with sync_playwright() as playwright:
            browser=playwright.chromium.launch(headless=True,args=['--no-sandbox'])
            page=browser.new_page(locale='es-PE',viewport={'width':1440,'height':1100});errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(f'http://127.0.0.1:{frontend.server_port}',wait_until='networkidle')
            page.screenshot(path=str(ROOT/'01-start.png'),full_page=True)
            page.locator('#pdf').set_input_files(str(pdf))
            page.wait_for_function("document.getElementById('doc').value.length>0 && !document.getElementById('claim').disabled",timeout=90000)
            document=json.loads(page.locator('#doc').input_value());lookup={p['id']:p for p in document['passages']}
            record['pdf_pages']=document['pages'];record['document_passages']=len(document['passages'])
            claims=[('de','Die schweizerische Währung ist der Franken.'),('fr','Le contre-projet direct propose d’inscrire dans la Constitution que la monnaie suisse est le franc.'),('it','Il controprogetto diretto stabilisce che la valuta svizzera è l’euro.')]
            for index,(language,claim) in enumerate(claims,1):
                page.locator('#claim').fill(claim)
                if index==1:page.screenshot(path=str(ROOT/'02-pdf-and-claim.png'),full_page=True)
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
                output=ROOT/f'prediction-{language}.json';download.value.save_as(str(output))
                assert json.loads(output.read_text())==result
                page.locator('#results').scroll_into_view_if_needed()
                page.evaluate("document.getElementById('results').scrollIntoView({block:'start'})")
                page.screenshot(path=str(ROOT/f'03-result-{language}.png'),full_page=False)
                record['predictions'].append({'claim_language':language,'claim':claim,'label':result['label'],'input_tokens':result['input_tokens'],'inference_time_ms':result['inference_time_ms'],'exact_source_quotes_pages_verified':True,'download_json_equal':True,'output':output.name})
                (ROOT/'verification.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n')
                print('LIVE_DEMO_REAL_BROWSER_PREDICTION',language,'label',result['label'],'tokens',result['input_tokens'],flush=True)
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
