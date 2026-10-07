"""Loopback native Apertus service; structured document inputs preserve context policy."""
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def make_server(engine, method='score', head=None, port=8001, pipeline=None):
    from ost_nli.v15 import REPO, REVISION
    if method=='head' and (not head or head.get('model_revision')!=REVISION):
        raise ValueError('A trained compatible v1.5 head is required')
    if method=='head' and head.get('feature_kind') not in ('hidden','option_logits'):
        raise ValueError('Paired research heads cannot be served by the single-forward endpoint')
    if pipeline and (method!='head' or pipeline.engine is not engine or pipeline.head!=head):
        raise ValueError('Document pipeline and serving engine/head must match')
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,obj):
            data=json.dumps(obj,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.reply(200,{'status':'ready','model':REPO,**engine.metadata,'method':method,'document_endpoint':pipeline is not None,'context_policy':pipeline.condition if pipeline else None}) if self.path=='/health' else self.reply(404,{'error':'Not found'})
        def do_POST(self):
            if self.path not in ('/v1/chat/completions','/nli'):self.reply(404,{'error':'Not found'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=8_000_000:raise ValueError('Invalid body size')
                body=json.loads(self.rfile.read(size))
                if self.path=='/nli':
                    if pipeline is None:raise ValueError('Document pipeline unavailable')
                    with lock:result=pipeline.predict(body['document'],body['claim'])
                    self.reply(200,result);return
                if pipeline and pipeline.condition!='full':
                    raise ValueError('Context-matched compact head requires booklet+claim through /nli')
                if body.get('model')!=REPO:raise ValueError('Wrong model')
                messages=body['messages']
                if not isinstance(messages,list) or not all(isinstance(x,dict) and isinstance(x.get('content'),str) for x in messages):raise ValueError('Invalid messages')
                with lock:result=engine.infer(messages,method=method,head=head)
                if result['invalid_output']:raise RuntimeError('Invalid model decision')
                self.reply(200,{'model':REPO,'choices':[{'message':{'role':'assistant','content':result['content']},'finish_reason':'stop'}],
                    'usage':{'prompt_tokens':result['context_tokens'],'completion_tokens':result['completion_tokens']},
                    'class_probabilities':result['probabilities'],'probability_note':result['probability_note'],
                    'decision_method':method,'model_revision':engine.metadata['revision']})
            except (ValueError,KeyError,TypeError):self.reply(400,{'error':'Invalid inference request'})
            except (RuntimeError,MemoryError):self.reply(503,{'error':'Apertus inference failed; no label produced'})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


