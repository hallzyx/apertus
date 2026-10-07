import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from test_core import example


class MockModel(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if hasattr(self.server, 'requests'):
            self.server.requests.append(payload)
        if self.path!='/v1/chat/completions' or payload['temperature'] != 0:
            self.send_error(400);return
        body=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"label":0}'}}],'usage':{'prompt_tokens':42}}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)


class CliIntegration(unittest.TestCase):
    def test_selected_full_context_reaches_model_through_cli_and_web(self):
        from ost_nli import web
        model = ThreadingHTTPServer(('127.0.0.1', 0), MockModel)
        model.requests = []
        app = ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (model, app)]
        for thread in threads: thread.start()
        doc = {'document_id': 'synthetic-context-check', 'passages': [
            {'id': 'background', 'page': 1, 'text': 'BACKGROUND_FULL_CONTEXT'},
            *[{'id': f'match-{i}', 'page': i+2, 'text': 'needle policy contribution'} for i in range(6)]
        ]}
        env = {'RETRIEVAL_MODE': 'full', 'LLM_BASE_URL': f'http://127.0.0.1:{model.server_port}/v1',
               'LLM_NAME': 'synthetic-mock-Apertus-v1.5', 'LLM_API_KEY': '',
               'OST_LABEL_MAP': '{"0":"entailment","1":"neutral","2":"contradiction"}'}
        try:
            with patch.dict(os.environ, env), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'document.json';path.write_text(json.dumps(doc))
                result = subprocess.run([sys.executable, '-m', 'ost_nli', 'predict', str(path), 'needle'],
                                        env=dict(os.environ), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                cli = json.loads(result.stdout)
                request = urllib.request.Request(f'http://127.0.0.1:{app.server_port}/api/predict',
                    data=json.dumps({'document': doc, 'claim': 'needle'}).encode(),
                    headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(request, timeout=10) as response:
                    live = json.load(response)
                for prediction in (cli, live):
                    self.assertEqual(prediction['retrieval'], 'full')
                    self.assertEqual(len(prediction['evidence']), 7)
                    self.assertFalse(prediction['truncated'])
                self.assertEqual(len(model.requests), 2)
                for request in model.requests:
                    self.assertIn('BACKGROUND_FULL_CONTEXT', request['messages'][1]['content'])
                    self.assertIn('match-5', request['messages'][1]['content'])
        finally:
            for server in (model, app): server.shutdown();server.server_close()
            for thread in threads: thread.join(timeout=2)

    def test_real_http_experiment_and_evaluator_roundtrip(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),MockModel)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);dataset=root/'synthetic.jsonl';registry=root/'synthetic-registry.jsonl';out=root/'results'
                dataset.write_text(''.join(json.dumps(example(i))+'\n' for i in range(3)))
                env={**os.environ,'LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','LLM_NAME':'synthetic-mock-Apertus-v1.5','LLM_API_KEY':'unit-test-not-a-real-key','OST_LABEL_MAP':'{"0":"entailment","1":"neutral","2":"contradiction"}'}
                run=subprocess.run([sys.executable,'-m','ost_nli','experiment',str(dataset),'--id','synthetic-only','--split-name','validation','--output-dir',str(out),'--registry',str(registry)],env=env,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
                report=json.loads(run.stdout);self.assertEqual(report['n'],3);self.assertEqual(report['average_context_tokens'],42)
                events=[json.loads(x) for x in registry.read_text().splitlines()]
                self.assertEqual([e['status'] for e in events],['started','completed'])
                self.assertNotIn('unit-test-not-a-real-key',registry.read_text())
                check=subprocess.run([sys.executable,'-m','ost_nli','evaluate',str(dataset),str(out/'predictions.json')],env=env,capture_output=True,text=True)
                self.assertEqual(check.returncode,0,check.stderr);self.assertEqual(json.loads(check.stdout),report)
                dataset.write_text(json.dumps(example(99))+'\n')
                mismatch=subprocess.run([sys.executable,'-m','ost_nli','evaluate',str(dataset),str(out/'predictions.json')],env=env,capture_output=True,text=True)
                self.assertNotEqual(mismatch.returncode,0)
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)
    def test_reference_not_misreported_as_full_booklet(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=example();row['document_scope']='provided_reference';dataset=Path(tmp)/'synthetic.jsonl';dataset.write_text(json.dumps(row)+'\n')
            run=subprocess.run([sys.executable,'-m','ost_nli','experiment',str(dataset),'--id','no-full-booklet','--split-name','validation','--context','full','--output-dir',str(Path(tmp)/'out')],capture_output=True,text=True)
            self.assertEqual(run.returncode,2);self.assertIn('Full-booklet baseline requires full booklets',run.stderr)
