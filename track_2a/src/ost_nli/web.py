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

HTML = r'''<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#f4f3ee"><title>Apertus · Evidence Lab</title>
<style>
:root{color-scheme:light;--paper:#f4f3ee;--ink:#172822;--muted:#54635b;--line:#d5dcd4;--red:#bc352c;--green:#1f5141;--soft:#e8ede6;--white:#fffefa}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 Arial,Helvetica,sans-serif}button,input,textarea{font:inherit}button,summary,.file-picker{position:relative;margin-top:14px}.file-picker input[type=file]{position:absolute;width:1px;height:1px;clip-path:inset(50%);overflow:hidden;white-space:nowrap}.file-picker label{display:inline-block;padding:8px 12px;border:1px solid var(--line);border-radius:4px;background:white;font-size:12px;font-weight:600;cursor:pointer}.file-picker:focus-within label{outline:3px solid #986900;outline-offset:4px}input[type=file]{cursor:pointer}button:disabled{cursor:not-allowed;opacity:.48}button:focus-visible,input:focus-visible,textarea:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #986900;outline-offset:4px}a{color:inherit}.skip{position:absolute;left:20px;top:-100px;z-index:10;background:white;padding:12px}.skip:focus{top:12px}.shell{width:min(1280px,100%);margin:auto;padding:0 40px}.masthead{display:flex;align-items:center;justify-content:space-between;padding:25px 0;border-bottom:1px solid var(--line);gap:20px}.brand{display:flex;align-items:center;gap:12px}.mark{width:34px;height:34px;background:var(--red);display:grid;place-items:center}.mark:after{content:'';width:14px;height:14px;border:3px solid var(--white)}.brand-name{font-size:21px;font-weight:800;letter-spacing:-.7px}.brand-sub{margin-left:12px;padding-left:16px;border-left:1px solid var(--line);font-size:13px;color:var(--muted)}.eyebrow{font:11px/1.4 'Courier New',monospace;font-weight:700;letter-spacing:1.5px;text-transform:uppercase}.header-note{color:var(--muted);text-align:right}.hero{display:grid;grid-template-columns:1.5fr 1fr;gap:60px;align-items:end;padding:48px 0 36px}.hero .eyebrow{color:var(--red);margin:0 0 15px}h1{font-size:clamp(36px,4.9vw,62px);line-height:1.04;letter-spacing:-2.8px;font-weight:500;margin:0}h1 span{color:var(--green)}.intro{max-width:430px;margin:0 0 18px;color:var(--muted);font-size:16px}.languages{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.language{font:12px 'Courier New',monospace;font-weight:bold;border:1px solid #b9c6ba;padding:5px 9px;border-radius:3px}.language-note{font-size:12px;color:var(--muted);margin-left:3px}.method-strip{display:flex;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:15px 0;margin-bottom:28px;gap:35px;flex-wrap:wrap}.method-strip span{font-size:12px;color:var(--muted)}.method-strip b{color:var(--green);font-family:'Courier New',monospace;margin-right:9px}.workspace{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.12fr);gap:24px;align-items:start}.panel{background:var(--white);border:1px solid var(--line);border-radius:8px;overflow:hidden}.panel-head{padding:22px 26px;border-bottom:1px solid var(--line);display:flex;gap:16px;justify-content:space-between;align-items:center}.panel-head{flex-wrap:wrap}.panel-head h2{font-size:18px;margin:0;font-weight:600;letter-spacing:-.4px}.panel-head .eyebrow{color:var(--muted);font-size:10px}.panel-body{padding:25px 26px}.step-label{display:flex;gap:9px;align-items:center;font-size:14px;font-weight:700;margin:0 0 10px}.step-number{color:var(--red);font:12px 'Courier New',monospace}.file-zone{padding:18px;border:1px dashed #9aaa9b;border-radius:5px;background:#f8f9f4}.file-zone strong{display:block;font-size:14px;margin-bottom:3px}.hint{font-size:12px;line-height:1.6;color:var(--muted);margin:5px 0 0}.file-picker{position:relative;margin-top:14px}.file-picker input[type=file]{position:absolute;width:1px;height:1px;clip-path:inset(50%);overflow:hidden;white-space:nowrap}.file-picker label{display:inline-block;padding:8px 12px;border:1px solid var(--line);border-radius:4px;background:white;font-size:12px;font-weight:600;cursor:pointer}.file-picker:focus-within label{outline:3px solid #986900;outline-offset:4px}input[type=file]{display:block;width:100%;max-width:100%;font-size:12px;color:var(--muted);margin:14px 0 0}input::file-selector-button{padding:8px 12px;border:1px solid var(--line);border-radius:4px;background:white;color:var(--ink);margin-right:10px;font:inherit;font-weight:600;cursor:pointer}.source{font-size:12px;color:var(--green);margin:10px 0 0;overflow-wrap:anywhere}.claim-field{margin-top:25px}textarea{width:100%;resize:vertical;min-height:116px;padding:14px;border:1px solid #a6b3a9;border-radius:5px;background:white;color:var(--ink);line-height:1.5}textarea::placeholder{color:#68736c}.field-meta{display:flex;justify-content:space-between;gap:12px}#claim-count{flex-shrink:0}.field-meta .hint{margin:3px 0 0}.actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px}button{border:1px solid var(--line);background:white;border-radius:5px;padding:11px 16px;font-weight:600;font-size:13px;color:var(--ink);transition:background .15s}button:hover:not(:disabled){background:var(--soft)}.primary{background:var(--green);color:white;border-color:var(--green)}.primary:hover:not(:disabled){background:#123e2e}.secondary{flex:1}.primary{flex:1}.backend{display:flex;gap:9px;margin-top:15px;font-size:12px;color:var(--muted)}.dot{height:7px;width:7px;flex-shrink:0;margin-top:6px;border-radius:50%;background:#8b7250}.backend[data-configured=true] .dot{background:#2d6d51}.advanced{border-top:1px solid var(--line);margin-top:22px;padding-top:16px}summary{font-size:12px;font-weight:600;padding:4px 0}details[open]>summary{margin-bottom:12px}.advanced textarea{font:12px/1.5 'Courier New',monospace;min-height:130px;margin:10px 0 3px}.advanced label{display:block;font-size:12px}.text-button{padding:7px 11px;font-size:12px;margin:10px 0 0}.result-panel{min-height:536px}.result-count{font:10px 'Courier New',monospace;background:var(--soft);padding:5px 8px;border-radius:3px;color:var(--green);white-space:nowrap}.empty{padding:62px 32px;text-align:center}.document-icon{width:52px;height:64px;border:1px solid #abb9ad;border-radius:3px;display:flex;flex-direction:column;justify-content:center;gap:8px;padding:12px;margin:0 auto 25px;transform:rotate(-5deg);box-shadow:6px 6px 0 var(--soft)}.document-icon i{display:block;height:1px;background:#92a496}.document-icon i:last-child{width:60%}.empty h3{font-size:23px;letter-spacing:-.6px;line-height:1.2;font-weight:500;margin:0 0 12px}.empty p{font-size:13px;color:var(--muted);max-width:330px;margin:0 auto}.empty .empty-label{margin-top:30px;font:10px 'Courier New',monospace;letter-spacing:1px;text-transform:uppercase;color:var(--muted)}#feedback{margin:0 0 18px;padding:12px 14px;background:var(--soft);border-radius:4px;font-size:13px}#feedback[data-kind=error]{color:#8d2823;background:#faece7;border-left:3px solid var(--red)}#feedback:empty{display:none}.result-content{padding:22px 26px}.result-content .empty{padding:40px 6px}.result-title{font-size:24px;line-height:1.2;letter-spacing:-.6px;margin:9px 0 12px;font-weight:600}.verdict-id{font:11px 'Courier New',monospace;color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:15px 0;margin:20px 0}.metric dt{font-size:10px;color:var(--muted);line-height:1.4}.metric dd{font:17px 'Courier New',monospace;margin:7px 0 0;font-weight:bold;overflow-wrap:anywhere}.result-note{font-size:12px;color:var(--muted);margin:0 0 15px}.warning{font-size:12px;padding:10px 12px;border-left:3px solid #987324;background:#f6f0df}.evidence-heading{display:flex;justify-content:space-between;align-items:center;margin:24px 0 12px}.evidence-heading h3{font-size:13px;margin:0}.passage{padding:15px 16px;margin:0 0 12px;background:#f6f7f2;border:1px solid #e0e5dc;border-radius:4px}.passage-meta{font:10px/1.4 'Courier New',monospace;color:var(--green);display:flex;justify-content:space-between;gap:10px;overflow-wrap:anywhere}.passage blockquote{font-size:14px;line-height:1.75;margin:10px 0 0;white-space:pre-wrap;overflow-wrap:anywhere}.raw{margin-top:20px;border-top:1px solid var(--line);padding-top:14px}.raw pre{font:11px/1.6 'Courier New',monospace;white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto;padding:12px;background:var(--soft)}.download{width:100%;margin-top:14px}.boundaries{display:grid;grid-template-columns:1fr 1fr;gap:32px;padding:26px 0 28px}.boundaries h2{font-size:12px;margin:0 0 5px}.boundaries p{font-size:12px;color:var(--muted);margin:0;max-width:550px}.benchmarks{font-size:12px;border-top:1px solid var(--line);padding:16px 0}.benchmarks p{color:var(--muted);overflow-wrap:anywhere}.benchmark-list{margin:12px 0;padding-left:20px;color:var(--muted)}footer{display:flex;justify-content:space-between;gap:15px;border-top:1px solid var(--line);padding:20px 0 30px;font:10px/1.5 'Courier New',monospace;color:var(--muted);letter-spacing:.5px}[hidden]{display:none!important}.spinner{display:inline-block;width:12px;height:12px;border:2px solid #aebcb0;border-top-color:var(--green);border-radius:50%;animation:spin 1s linear infinite;margin-right:8px;vertical-align:-2px}@keyframes spin{to{transform:rotate(360deg)}}@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}@media(max-width:850px){.shell{padding:0 24px}.hero{gap:28px;grid-template-columns:1fr 1fr}h1{font-size:46px}.workspace{grid-template-columns:1fr}.result-panel{min-height:350px}.empty{padding:40px 25px}.hero .intro{font-size:14px}}@media(max-width:540px){.shell{padding:0 18px}.masthead{padding:18px 0}.brand-sub{font-size:11px;padding-left:10px;margin-left:4px}.header-note{display:none}.hero{grid-template-columns:1fr;padding:32px 0 25px;gap:20px}h1{font-size:43px;letter-spacing:-2px}.intro{max-width:none!important;margin-bottom:14px}.method-strip{gap:12px 20px;padding:12px 0;margin-bottom:20px}.method-strip span{font-size:10px}.method-strip b{margin-right:5px}.panel-head{padding:18px}.panel-body,.result-content{padding:20px 18px}.boundaries{grid-template-columns:1fr;gap:18px}.actions{flex-direction:column}.actions button{width:100%}.metrics{gap:8px}.metric dd{font-size:15px}footer{flex-direction:column;gap:6px}}
</style>
</head>
<body>
<a class="skip" href="#workspace">Ir al laboratorio</a>
<div class="shell">
<header class="masthead"><div class="brand"><span class="mark" aria-hidden="true"></span><span class="brand-name">apertus</span><span class="brand-sub">Evidence Lab</span></div><span class="header-note eyebrow">Swiss multilingual NLI<br>Track 2A · OST</span></header>
<main>
<section class="hero" aria-labelledby="title"><div><p class="eyebrow">Lectura crítica · Evidencia documental</p><h1 id="title">Del documento<br>a la <span>evidencia.</span></h1></div><div><p class="intro">Contrasta una afirmación con un folleto de votación suizo. Explora los pasajes de origen y revisa lo que el documento permite concluir.</p><div class="languages" aria-label="Idiomas admitidos"><span class="language" lang="de">DE</span><span class="language" lang="fr">FR</span><span class="language" lang="it">IT</span><span class="language-note">Tres idiomas. Una fuente.</span></div></div></section>
<div class="method-strip" aria-label="Pasos del análisis"><span><b>01</b> Carga el documento</span><span><b>02</b> Formula una afirmación</span><span><b>03</b> Revisa la evidencia</span></div>
<div class="workspace" id="workspace">
<section class="panel" aria-labelledby="input-title"><div class="panel-head"><h2 id="input-title">Tu consulta</h2><span class="eyebrow">Documento + afirmación</span></div><div class="panel-body">
<form id="analysis-form">
<label class="step-label" for="pdf"><span class="step-number">01 /</span> Folleto de votación</label>
<div class="file-zone"><strong>Empieza con el documento original</strong><p class="hint" id="pdf-help">PDF con texto seleccionable · hasta 50 MB.<br>La extracción conserva las páginas de origen.</p><div class="file-picker"><input id="pdf" type="file" accept="application/pdf,.pdf" aria-describedby="pdf-help source"><label for="pdf">Seleccionar PDF</label></div></div>
<p class="source" id="source" role="status">Ningún documento seleccionado.</p>
<div class="claim-field"><label class="step-label" for="claim"><span class="step-number">02 /</span> Afirmación a contrastar</label><textarea id="claim" maxlength="16000" required placeholder="Escribe una afirmación concreta en alemán, francés o italiano…" aria-describedby="claim-help"></textarea><div class="field-meta"><p class="hint" id="claim-help">Puede estar en un idioma distinto al del documento.</p><p class="hint" id="claim-count">0 / 16 000</p></div></div>
<div class="actions"><button class="secondary" id="retrieve" type="button" disabled>Inspeccionar pasajes</button><button class="primary" id="predict" type="submit" disabled>Evaluar afirmación <svg aria-hidden="true" width="13" height="13" viewBox="0 0 16 16"><path d="M3 13 13 3M3 3h10v10" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></button></div>
<div class="backend" id="backend"><span class="dot" aria-hidden="true"></span><span id="status" role="status">Comprobando configuración del backend…</span></div>
<details class="advanced"><summary>Entrada avanzada / ejemplo sintético</summary><p class="hint">Puedes pegar un documento canónico en JSON. El ejemplo es inventado y solo sirve para explorar la interfaz.</p><button type="button" id="demo" class="text-button">Cargar ejemplo sintético</button><label for="doc" class="hint">JSON con document_id y passages</label><textarea id="doc" spellcheck="false" placeholder='{"document_id":"…","passages":[{"id":"p1","page":1,"text":"…"}]}'></textarea><p class="hint">Editar esta entrada sustituye al PDF seleccionado.</p></details>
</form></div></section>
<section class="panel result-panel" aria-labelledby="result-heading" aria-busy="false" id="results"><div class="panel-head"><h2 id="result-heading" tabindex="-1">Evidencia documental</h2><span class="result-count" id="result-state">POR EXPLORAR</span></div><div class="result-content"><div id="feedback" role="status" aria-live="polite" tabindex="-1"></div><div id="summary"><div class="empty"><div class="document-icon" aria-hidden="true"><i></i><i></i><i></i></div><h3>La fuente es el punto<br>de partida.</h3><p>Carga un folleto y escribe una afirmación para ver los pasajes relevantes, con sus páginas de origen.</p><p class="empty-label">Cada conclusión requiere lectura crítica</p></div></div><button type="button" class="download" id="download" hidden>Descargar resultado JSON <svg aria-hidden="true" width="13" height="13" viewBox="0 0 16 16"><path d="M8 2v10m-4-4 4 4 4-4M2 14h12" fill="none" stroke="currentColor" stroke-width="1.5"/></svg></button><details class="raw" id="raw" hidden><summary>Ver respuesta completa · JSON</summary><pre id="result"></pre></details></div></section>
</div>
<section class="boundaries" aria-label="Cómo interpretar los resultados"><div><h2>El documento define el alcance.</h2><p>La clasificación describe la relación entre la afirmación y el folleto. No determina la verdad general ni ofrece consejo político.</p></div><div><h2>Los pasajes requieren tu revisión.</h2><p>Inspeccionar pasajes permite explorar la fuente sin un backend de inferencia. Evaluar requiere Apertus; los pasajes recuperados no son una conclusión.</p></div></section>
<details class="benchmarks"><summary>Resultados del laboratorio y alcance de las métricas</summary><div id="benchmarks"><p>Cargando resultados registrados…</p></div></details>
</main>
<footer><span>APERTUS EVIDENCE LAB · OST / TRACK 2A</span><span>Herramienta de investigación · No es un servicio oficial de votación</span></footer>
</div>
<script>
'use strict';
const byId=id=>document.getElementById(id);
const status=byId('status'),result=byId('result'),summary=byId('summary'),download=byId('download');
let latestResult=null,busy=false,modelConfigured=false,sourceKind=null;
const emptyMarkup=summary.innerHTML;
function element(tag,text,className){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;}
function refresh(){const ready=Boolean(byId('doc').value.trim()&&byId('claim').value.trim());byId('retrieve').disabled=busy||!ready;byId('predict').disabled=busy||!ready||!modelConfigured;for(const id of ['pdf','claim','doc','demo'])byId(id).disabled=busy;byId('results').setAttribute('aria-busy',String(busy));}
function feedback(message,kind='info'){const box=byId('feedback');box.replaceChildren();box.dataset.kind=kind;if(busy){const spinner=element('span',undefined,'spinner');spinner.setAttribute('aria-hidden','true');box.append(spinner);}box.append(document.createTextNode(message));}
function invalidate(){latestResult=null;download.hidden=true;byId('raw').hidden=true;byId('raw').open=false;result.textContent='';summary.innerHTML=emptyMarkup;byId('result-state').textContent='POR EXPLORAR';feedback('');}
function formatNumber(n){return typeof n==='number'&&Number.isFinite(n)?n.toLocaleString('es-ES',{maximumFractionDigits:0}):'No disponible';}
function showResult(data,action){summary.replaceChildren();const isPrediction=action==='predict'&&Number.isInteger(data.label)&&data.label>=0&&data.label<=2;const evidence=Array.isArray(data.evidence)?data.evidence:[];
 byId('result-state').textContent=isPrediction?'EVALUACIÓN':'PASAJES';
 summary.append(element('p',sourceKind==='synthetic'?'EJEMPLO SINTÉTICO · NO ES UN FOLLETO OFICIAL':'FUENTE · '+byId('source').textContent,'verdict-id'));
 if(isPrediction){const titles=['Apoyada por el documento','Información insuficiente','Contradicha por el documento'];summary.append(element('h3',titles[data.label],'result-title'),element('p',['0 · entailment','1 · neutral','2 · contradiction'][data.label]+' · Clasificación del modelo','verdict-id'));
 const metrics=element('dl',undefined,'metrics');const p=data.probabilities;const probability=Array.isArray(p)&&p.length===3&&p.every(v=>typeof v==='number'&&Number.isFinite(v)&&v>=0&&v<=1)?(p[data.label]*100).toLocaleString('es-ES',{maximumFractionDigits:1})+' %':'No disponible';
 for(const [label,value] of [['Probabilidad de clase¹',probability],['Tokens de entrada',formatNumber(data.input_tokens??data.context_tokens)],['Tiempo de inferencia',typeof data.inference_time_ms==='number'?formatNumber(data.inference_time_ms)+' ms':'No disponible']]){const metric=element('div',undefined,'metric');metric.append(element('dt',label),element('dd',value));metrics.append(metric);}summary.append(metrics,element('p','¹ Probabilidad producida por el modelo, sin garantía de calibración. No mide la veracidad de una afirmación.','result-note'));
 }else{summary.append(element('h3','Pasajes para tu revisión','result-title'),element('p','La recuperación selecciona texto relacionado con la consulta. No clasifica la afirmación como apoyada o contradicha.','result-note'));}
 if(data.truncated)summary.append(element('p','Contexto truncado: el modelo no ha recibido todo el texto disponible. Revisa el alcance antes de interpretar la clasificación.','warning'));
 if(data.evidence_note)summary.append(element('p',data.evidence_note,'result-note'));
 if(isPrediction)summary.append(element('p','Estos pasajes acompañan la respuesta del modelo; revisa su relevancia y suficiencia en el documento original.','result-note'));
 const heading=element('div',undefined,'evidence-heading');heading.append(element('h3','Pasajes del documento'),element('span',String(evidence.length)+(evidence.length===1?' pasaje':' pasajes'),'verdict-id'));summary.append(heading);
 if(!evidence.length)summary.append(element('p','La respuesta no contiene pasajes. No se han generado citas.','result-note'));
 for(const passage of evidence){const card=element('article',undefined,'passage');const meta=element('div',undefined,'passage-meta');meta.append(element('span',passage.page==null?'PÁGINA NO DISPONIBLE':'PÁGINA '+passage.page),element('span',passage.id==null?'':String(passage.id)));const quote=element('blockquote',typeof passage.text==='string'?passage.text:'Texto no disponible');quote.setAttribute('dir','auto');meta.setAttribute('aria-label','Referencia del pasaje');card.append(meta,quote);summary.append(card);}
}
async function request(url,options={},timeout=120000){const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),timeout);try{const response=await fetch(url,{...options,signal:controller.signal});let data;try{data=await response.json();}catch{throw Error('El servidor devolvió una respuesta que no se pudo leer. Inténtalo de nuevo.');}if(!response.ok){if(url==='/api/document')throw Error('No se pudo extraer texto del PDF. Usa un archivo de hasta 50 MB, sin cifrar y con texto seleccionable.');if(response.status===502)throw Error('No se pudo conectar con Apertus. No se ha producido ninguna decisión. Comprueba el backend e inténtalo de nuevo.');throw Error('La solicitud no se pudo completar. Revisa el documento, la afirmación y la configuración del modelo.');}return data;}catch(error){if(error.name==='AbortError')throw Error('La solicitud tardó demasiado. Comprueba el servidor e inténtalo de nuevo.');if(error instanceof TypeError)throw Error('No se pudo conectar con el servidor. Comprueba la conexión e inténtalo de nuevo.');throw error;}finally{clearTimeout(timer);}}
async function run(action){if(busy||(action==='predict'&&!modelConfigured))return;invalidate();let documentData;try{documentData=JSON.parse(byId('doc').value);if(!documentData||typeof documentData.document_id!=='string'||!Array.isArray(documentData.passages)||!documentData.passages.length)throw Error();}catch{feedback('El documento JSON no es válido. Incluye document_id y una lista de passages con texto.','error');byId('feedback').focus();return;}if(!byId('claim').value.trim()){byId('claim').focus();return;}
 busy=true;refresh();byId('result-state').textContent='EN CURSO';feedback(action==='predict'?'Evaluando la afirmación con Apertus…':'Buscando pasajes en el documento…');
 try{const data=await request('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({document:documentData,claim:byId('claim').value.trim()})});if(action==='predict'&&(!Number.isInteger(data.label)||data.label<0||data.label>2))throw Error('El backend no devolvió una clasificación válida. No se ha mostrado ninguna decisión.');latestResult=data;showResult(data,action);result.textContent=JSON.stringify(data,null,2);download.hidden=false;byId('raw').hidden=false;busy=false;feedback(action==='predict'?'Evaluación completada. Revisa la evidencia.':'Pasajes recuperados. No se ha emitido una clasificación.');byId('result-heading').focus();}catch(error){busy=false;feedback(error.message,'error');byId('result-state').textContent='REVISAR';byId('feedback').focus();}finally{busy=false;refresh();}}
byId('analysis-form').onsubmit=event=>{event.preventDefault();run('predict');};byId('retrieve').onclick=()=>run('retrieve');
byId('claim').oninput=()=>{invalidate();byId('claim-count').textContent=byId('claim').value.length.toLocaleString('es-ES')+' / 16 000';refresh();};
byId('doc').oninput=()=>{sourceKind='json';byId('pdf').value='';byId('source').textContent=byId('doc').value.trim()?'Documento JSON introducido manualmente.':'Ningún documento seleccionado.';invalidate();refresh();};
byId('demo').onclick=()=>{sourceKind='synthetic';byId('pdf').value='';byId('doc').value=JSON.stringify({document_id:'synthetic-demo-only',passages:[{id:'p1',page:1,text:'Die Vorlage sieht einen jährlichen Beitrag von 100 Franken vor.'},{id:'p2',page:2,text:'La proposition prévoit une contribution annuelle de 100 francs.'},{id:'p3',page:3,text:'La proposta prevede un contributo annuo di 100 franchi.'}]},null,2);byId('claim').value='Die Vorlage sieht einen jährlichen Beitrag von 100 Franken vor.';byId('claim-count').textContent=byId('claim').value.length+' / 16 000';byId('source').textContent='Ejemplo sintético · 3 pasajes · no es un folleto oficial';invalidate();refresh();};
byId('pdf').onchange=async()=>{const file=byId('pdf').files[0];if(!file)return;invalidate();byId('doc').value='';sourceKind=null;byId('source').textContent='Leyendo '+file.name+'…';if(file.size>50000000){byId('source').textContent='Archivo no cargado.';feedback('El PDF supera el límite de 50 MB. Selecciona un archivo más pequeño.','error');refresh();return;}busy=true;refresh();feedback('Extrayendo texto y páginas del PDF…');try{const documentData=await request('/api/document',{method:'POST',headers:{'Content-Type':'application/pdf'},body:file});byId('doc').value=JSON.stringify(documentData);sourceKind='pdf';byId('source').textContent=file.name+' · '+(documentData.pages??'—')+(documentData.pages===1?' página · ':' páginas · ')+documentData.passages.length+(documentData.passages.length===1?' pasaje':' pasajes');busy=false;feedback('Documento listo. Escribe la afirmación que quieres contrastar.');byId('claim').focus();}catch(error){busy=false;byId('source').textContent='Archivo no cargado. Selecciona otro PDF.';byId('pdf').value='';feedback(error.message,'error');byId('feedback').focus();}finally{busy=false;refresh();}};
download.onclick=()=>{if(!latestResult)return;const url=URL.createObjectURL(new Blob([JSON.stringify(latestResult,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='apertus-result.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
request('/health',{},15000).then(health=>{modelConfigured=health.model_configured===true;byId('backend').dataset.configured=String(modelConfigured);status.textContent=modelConfigured?'Backend configurado. La conexión se comprueba al evaluar.':'Sin backend de Apertus. Puedes inspeccionar pasajes; la evaluación requiere el modelo.';refresh();}).catch(()=>{status.textContent='No se pudo comprobar el backend. Recarga la página para reintentar; puedes inspeccionar pasajes.';refresh();});
request('/benchmarks',{},15000).then(data=>{const box=byId('benchmarks');box.replaceChildren();if(!Array.isArray(data.results)||!data.results.length){box.append(element('p','No hay métricas registradas disponibles.'));return;}const list=element('ul',undefined,'benchmark-list');for(const row of data.results){if(typeof row.macro_f1==='number')list.append(element('li',row.experiment+' · '+(row.split||'histórico')+' · Macro-F1 '+row.macro_f1.toFixed(4)));}box.append(list,element('p',data.scope||'Las métricas corresponden a experimentos registrados; no garantizan el resultado de una consulta nueva.'));}).catch(()=>{byId('benchmarks').replaceChildren(element('p','Las métricas registradas no están disponibles en este momento.'));});
</script>
</body></html>
'''


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
