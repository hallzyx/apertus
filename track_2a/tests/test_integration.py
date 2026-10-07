import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from test_core import example


class MockModel(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
        payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path!='/v1/chat/completions' or payload['temperature'] != 0:
            self.send_error(400);return
        body=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"label":0}'}}],'usage':{'prompt_tokens':42}}).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)


class CliIntegration(unittest.TestCase):
    def test_real_http_experiment_and_evaluator_roundtrip(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),MockModel)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);dataset=root/'synthetic.jsonl';registry=root/'synthetic-registry.jsonl';out=root/'results'
                dataset.write_text(''.join(json.dumps(example(i))+'\n' for i in range(3)))
                env={**os.environ,'LLM_BASE_URL':f'http://127.0.0.1:{server.server_port}/v1','LLM_NAME':'synthetic-mock-Apertus','LLM_API_KEY':'unit-test-not-a-real-key','OST_LABEL_MAP':'{"0":"entailment","1":"contradiction","2":"neutral"}'}
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
