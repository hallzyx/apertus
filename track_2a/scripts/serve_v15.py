"""Single-device, loopback-only Apertus v1.5 chat endpoint for the public CLI."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def make_server(engine, method='score', head=None, port=8001):
    from ost_nli.v15 import REPO, REVISION
    if method=='head' and (not head or head.get('model_revision')!=REVISION):
        raise ValueError('A trained compatible v1.5 head is required')
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,obj):
            data=json.dumps(obj,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            self.reply(200,{'status':'ready','model':REPO,**engine.metadata,'method':method}) if self.path=='/health' else self.reply(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/v1/chat/completions':self.reply(404,{'error':'Not found'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2_000_000:raise ValueError('Invalid body size')
                body=json.loads(self.rfile.read(size))
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


def main():
    from ost_nli.v15 import Engine
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--head')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--method',choices=['score','prompt','head'],default='score')
    p.add_argument('--port',type=int,default=8001)
    a=p.parse_args();engine=Engine(a.model_dir,a.device)
    head=json.loads(open(a.head).read()) if a.head else None
    server=make_server(engine,a.method,head,a.port)
    print('V15_SERVING_READY',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
