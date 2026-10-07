"""Contract tests use a fake encoder; they assert no NLI accuracy."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from ost_nli.context_budget import DocumentPipeline
from ost_nli.v15 import REPO,REVISION


class Tokenizer:
    def apply_chat_template(self,messages,**kwargs):return json.dumps(messages,ensure_ascii=False)
    def __call__(self,text,**kwargs):return {'input_ids':text.split()}


class Encoder:
    tokenizer=Tokenizer()
    metadata={'repo':REPO,'revision':REVISION,'precision':'FAKE CONTRACT TEST ONLY'}
    def __init__(self):self.inputs=[]
    def infer(self,messages,**kwargs):
        self.inputs.append(messages)
        return {'label':0,'invalid_output':False,'probabilities':[.8,.1,.1],
                'probability_note':'FAKE CONTRACT TEST ONLY','context_tokens':len((self.tokenizer.apply_chat_template(messages)+'{"label":').split()),'latency_seconds':.01}


class Retriever:
    def retrieve(self,passages,claim,k,mode):return passages[:k]


class ContextServingTests(unittest.TestCase):
    def test_compact_pipeline_preserves_quotes_and_never_uses_reference(self):
        engine=Encoder();head={'context':'hybrid-1k','feature_kind':'hidden','model_revision':REVISION}
        pipeline=DocumentPipeline(engine,head,Retriever())
        doc={'document_id':'contract-test','passages':[{'id':'oversize','page':1,'text':'large '*1200},
             {'id':'exact','page':7,'text':'La proposta prevede 100 franchi.','char_start':12,'char_end':44}],
             'reference_string':'GOLD_MARKER_MUST_NOT_BE_FORWARDED','label':2}
        result=pipeline.predict(doc,'Die Vorlage sieht 100 Franken vor.')
        self.assertEqual(result['evidence'][0]['text'],doc['passages'][1]['text'])
        self.assertEqual(result['evidence'][0]['page'],7)
        self.assertLessEqual(result['input_tokens'],1024)
        self.assertNotIn('GOLD_MARKER',json.dumps(engine.inputs))
        with self.assertRaises(ValueError):pipeline.predict([], 'claim')
        with self.assertRaises(ValueError):DocumentPipeline(engine,head)

    def test_document_service_reaches_public_single_and_batch_cli(self):
        from ost_nli.native_service import make_server
        engine=Encoder();head={'context':'hybrid-1k','feature_kind':'hidden','model_revision':REVISION}
        server=make_server(engine,'head',head,0,DocumentPipeline(engine,head,Retriever()))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            body={'model':REPO,'messages':[{'role':'user','content':'Unpacked arbitrary text'}]}
            req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
            self.assertEqual(error.exception.code,400);self.assertEqual(engine.inputs,[])
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);booklet=root/'booklet.json'
                booklet.write_text(json.dumps({'document_id':'test','passages':[{'id':'p','page':3,'text':'Exact source quote.'}]}))
                batch=root/'batch.jsonl';batch.write_text(json.dumps({'id':'one','document':'booklet.json','claim':'A claim'})+'\n')
                environment={**os.environ,'FROZEN_BASE_URL':base,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
                for command in [['predict',str(booklet),'A claim'],['predict-batch',str(batch),'--output',str(root/'out.json')]]:
                    run=subprocess.run([sys.executable,'-m','ost_nli',*command],env=environment,capture_output=True,text=True)
                    self.assertEqual(run.returncode,0,run.stderr)
                result=json.loads((root/'out.json').read_text())['predictions'][0]
                self.assertEqual(result['id'],'one');self.assertEqual(result['evidence'][0]['page'],3)
                self.assertEqual(result['context_policy'],'hybrid-1k')
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
