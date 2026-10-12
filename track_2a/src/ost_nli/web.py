import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .data import validate_document
from .model import ApertusClient, FrozenServiceClient, label_map, predict, select_context
from .retrieval import retrieve
from .runtime import production_context, document_service_url

RETRIEVER = None

def selected_passages(document, claim):
    mode = production_context()
    if mode == 'full':
        return [{**p, 'retrieval_method': 'full'} for p in select_context(document, mode='full')[0]]
    if mode == 'bm25' or RETRIEVER is None:
        return retrieve(document['passages'], claim, k=5)
    return RETRIEVER.retrieve(document['passages'], claim, k=5,
                              mode=mode)

HTML = r'''<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Apertus · Evidence Lab</title>
<style>body{font:16px system-ui;background:#101827;color:#e8eef8;max-width:1050px;margin:50px auto;padding:0 24px}h1{font-size:42px}p{line-height:1.6;color:#b5c2d6}textarea,input,select{box-sizing:border-box;width:100%;background:#19253a;color:#fff;border:1px solid #43536d;border-radius:8px;padding:12px;margin:8px 0}textarea{min-height:160px}button{background:#77e3c2;color:#10231c;border:0;border-radius:8px;padding:12px 22px;font-weight:700;margin:12px 12px 12px 0;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#19253a;padding:20px;border-radius:12px}label{display:block;margin-top:20px}.tag{color:#77e3c2}#status{padding:12px;border:1px solid #43536d;border-radius:8px}small{color:#b5c2d6}</style>
<span class="tag">TRACK 2A · OST · RESEARCH WORKBENCH</span><h1>Apertus Evidence Lab</h1><p>Evalúa afirmaciones contra un folleto oficial de votación suizo. La app devuelve pasajes y páginas del documento, tokens y tiempo. El resultado describe lo que apoya el folleto; no determina la verdad general ni ofrece consejo político.</p>
<details><summary>Resultados del laboratorio</summary><div id="benchmarks"></div></details><div id="status">Comprobando configuración…</div>
<details><summary>Entrada JSON avanzada / ejemplo sintético</summary><label for="doc">Booklet (canonical JSON with document_id and passages)</label><textarea id="doc">{"document_id":"synthetic-demo-only","passages":[{"id":"p1","page":1,"text":"Die Vorlage sieht einen jährlichen Beitrag von 100 Franken vor."},{"id":"p2","page":2,"text":"La proposition prévoit une contribution annuelle de 100 francs."},{"id":"p3","page":3,"text":"La proposta prevede un contributo annuo di 100 franchi."}]}</textarea>
<small>This example is synthetic, not an official voting booklet. Carga tu PDF abajo para evaluar un folleto real.</small>
</details><label for="pdf">1. Sube el folleto oficial (PDF con texto)</label><input id="pdf" type="file" accept="application/pdf">
<label for="claim">2. Escribe la afirmación · DE / FR / IT</label><input id="claim" value="Die Vorlage sieht einen jährlichen Beitrag von 100 Franken vor.">
<button id="retrieve">Inspeccionar pasajes</button><button id="predict">Evaluar afirmación</button><button id="download" hidden>Descargar resultado JSON</button><pre id="result" aria-live="polite">Sube un PDF y escribe una afirmación. Puedes inspeccionar pasajes sin backend; la evaluación requiere Apertus.</pre>
<script>
const status=document.getElementById('status'),result=document.getElementById('result');
let latestResult=null;const download=document.getElementById('download');download.onclick=()=>{if(!latestResult)return;const url=URL.createObjectURL(new Blob([JSON.stringify(latestResult,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='apertus-result.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
const summary=document.createElement('section');summary.setAttribute('aria-live','polite');result.before(summary);
function showPrediction(data){summary.replaceChildren();if(!Number.isInteger(data.label)||data.label<0||data.label>2)return;
 const verdict=document.createElement('h2');verdict.textContent=['0 · Apoyada · Entailment','1 · Información insuficiente · Neutral','2 · Contradicha · Contradiction'][data.label];summary.append(verdict);
 const usage=document.createElement('p'),p=data.probabilities;
 usage.textContent='Confidence: '+(Array.isArray(p)?(100*p[data.label]).toFixed(1)+'%':'unavailable')+' · Input tokens: '+(data.input_tokens??data.context_tokens??'unavailable')+' · Time: '+(typeof data.inference_time_ms==='number'?data.inference_time_ms.toFixed(0)+' ms':'unavailable');summary.append(usage);
 const note=document.createElement('p');note.textContent='Confidence describes the model decision, not political truth. '+(data.evidence_note||'The passages below were supplied to the model; their relevance and sufficiency need review.');summary.append(note);
 for(const passage of data.evidence||[]){const quote=document.createElement('blockquote');quote.textContent='Page '+(passage.page??'unknown')+' · '+passage.text;summary.append(quote);}
}
fetch('/benchmarks').then(r=>r.json()).then(b=>{if(b.results){document.getElementById('benchmarks').textContent='Recorded Macro-F1 · '+b.results.map(r=>r.experiment+' ['+(r.split||'historical')+']: '+r.macro_f1.toFixed(4)).join(' · ')+' · '+b.scope;}});
fetch('/health').then(r=>r.json()).then(h=>{status.textContent=h.model_configured?'Backend configurado · realiza una evaluación para comprobar la conexión.':'Sin backend de Apertus · puedes leer el PDF e inspeccionar pasajes; la clasificación está deshabilitada.';document.getElementById('predict').disabled=!h.model_configured;});
async function run(action){let button=document.getElementById(action);button.disabled=true;summary.replaceChildren();try{const body={document:JSON.parse(document.getElementById('doc').value),claim:document.getElementById('claim').value};latestResult=null;download.hidden=true;result.textContent='Evaluando…';const response=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await response.json();latestResult=response.ok?data:null;download.hidden=!response.ok;if(response.ok&&action==='predict')showPrediction(data);result.textContent=JSON.stringify(data,null,2);}catch(e){result.textContent='Invalid input or network failure. Please check the booklet JSON.';}finally{button.disabled=false;}}
document.getElementById('retrieve').onclick=()=>run('retrieve');document.getElementById('predict').onclick=()=>run('predict');
document.getElementById('pdf').onchange=async()=>{const file=document.getElementById('pdf').files[0];if(!file)return;latestResult=null;download.hidden=true;summary.replaceChildren();result.textContent='Leyendo el PDF…';try{const r=await fetch('/api/document',{method:'POST',headers:{'Content-Type':'application/pdf'},body:file});const d=await r.json();if(!r.ok)throw Error();document.getElementById('doc').value=JSON.stringify(d);result.textContent='PDF cargado: '+d.pages+' páginas. Las citas conservan la página de origen.';}catch(e){result.textContent='No se pudo leer el PDF. Usa un PDF con texto seleccionable.';}};
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Do not log request bodies, claims, keys or endpoint errors.
        pass

    def respond(self,status,body,content_type="application/json"):
        payload = body.encode() if isinstance(body,str) else json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type",content_type+"; charset=utf-8")
        self.send_header("Content-Length",str(len(payload)))
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            self.respond(200,HTML,"text/html")
        elif self.path == "/benchmarks":
            path = Path(__file__).resolve().parents[2]/"deployment/benchmark.json"
            self.respond(200,json.loads(path.read_text()) if path.exists() else {"available":False})
        elif self.path == "/health":
            configured = False
            try:
                service=document_service_url(strict=True)
                (FrozenServiceClient(service) if service else (ApertusClient(),label_map()));configured = True
            except (ValueError,TypeError):
                pass
            self.respond(200,{"status":"ok","retrieval_available":True,"model_configured":configured,"model_connectivity_verified":False,"benchmark_verified":False})
        else:
            self.respond(404,{"error":"Not found"})

    def do_POST(self):
        if self.path == '/api/document':
            import tempfile
            try:
                size = int(self.headers.get('Content-Length','0'))
                if size < 1 or size > 50_000_000:
                    self.respond(413,{'error':'PDF must be at most 50 MB'});return
                content = self.rfile.read(size)
                if not content.startswith(b'%PDF-'): raise ValueError('Invalid PDF')
                from .documents import pdf_document
                with tempfile.NamedTemporaryFile(suffix='.pdf') as file:
                    file.write(content);file.flush();document=pdf_document(file.name)
                self.respond(200,document)
            except Exception:
                self.respond(400,{'error':'Cannot extract source text from PDF; encrypted or scanned PDFs need separate processing'})
            return
        if self.path not in ("/api/retrieve","/api/predict"):
            self.respond(404,{"error":"Not found"});return
        try:
            size = int(self.headers.get("Content-Length","0"))
            if size < 1 or size > 2_000_000:
                self.respond(413,{"error":"Require a JSON body up to 2 MB"});return
            payload = json.loads(self.rfile.read(size))
            doc = validate_document(payload["document"])
            claim = payload["claim"]
            if not isinstance(claim,str) or not claim.strip() or len(claim)>16000:
                raise ValueError("Claim must be nonempty and at most 16000 characters")
            if self.path == "/api/retrieve":
                passages = selected_passages(doc,claim)
                result = {"evidence":passages,"method":passages[0].get('retrieval_method','bm25')}
            else:
                service=document_service_url(strict=True)
                if service:
                    result = FrozenServiceClient(service).predict(doc,claim)
                else:
                    import time
                    began = time.perf_counter()
                    mode = production_context()
                    if mode == 'full':
                        result = predict({**doc,"claim":claim},ApertusClient(),label_map(),mode='full')
                    else:
                        passages = selected_passages(doc,claim)
                        result = predict({**doc,'passages':passages,"claim":claim},ApertusClient(),label_map(),mode='full')
                    result['retrieval'] = mode
                    result['model_inference_time_ms']=result['inference_time_ms']
                    result['inference_time_ms']=(time.perf_counter()-began)*1000
                    result['latency_seconds']=result['inference_time_ms']/1000
            self.respond(200,result)
        except (ValueError,KeyError,TypeError):
            self.respond(400,{"error":"Invalid input or missing model/official label configuration"})
        except RuntimeError:
            self.respond(502,{"error":"Apertus request failed; no decision produced"})


def serve(host,port):
    server = ThreadingHTTPServer((host,port),Handler)
    print(f"Evidence Lab listening on port {port}",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
