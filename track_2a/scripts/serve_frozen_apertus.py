"""Serve the real frozen Apertus classifier for the lightweight Docker workbench."""
import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ost_nli.data import validate_document
from ost_nli.frozen import FrozenApertus
from ost_nli.model import label_map


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--head-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8001)
    args=p.parse_args()
    mapping=label_map() if os.environ.get('OST_LABEL_MAP') else None
    backend=FrozenApertus(args.model_dir,args.head_dir,args.device)
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,body):
            payload=json.dumps(body,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):
            if self.path=='/health':
                self.reply(200,{'status':'ready','model':backend.experiment['model'],
                    'model_revision':backend.experiment['model_revision'],
                    'head_experiment':backend.experiment['experiment_id'],'device':args.device,
                    'official_class_semantics_verified':mapping is not None})
            else:self.reply(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/nli':self.reply(404,{'error':'Not found'});return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if size<1 or size>2_000_000:self.reply(413,{'error':'Require JSON up to 2 MB'});return
                request=json.loads(self.rfile.read(size));document=validate_document(request['document'])
                with lock:
                    result=backend.predict(document,request['claim'],request.get('context','full'),request.get('k',5),mapping)
                self.reply(200,result)
            except (ValueError,KeyError,TypeError):self.reply(400,{'error':'Invalid document, claim or inference options'})
            except (RuntimeError,MemoryError):self.reply(503,{'error':'Inference failed; no decision produced'})
    print('Verified frozen Apertus backend ready on port',args.port,flush=True)
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()


if __name__=='__main__':main()
