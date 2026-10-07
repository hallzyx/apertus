"""Checkpoint failure recovery with a fake encoder; no real model result claimed."""
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Phase2ResumeTests(unittest.TestCase):
    def test_paired_research_head_cannot_silently_use_single_forward_features(self):
        from ost_nli.v15 import Engine
        engine = object.__new__(Engine)
        with self.assertRaisesRegex(ValueError,'explicit paired runtime'):
            engine.infer([],method='head',head={'feature_kind':'paired_hidden_difference'})

    def test_completed_chunks_are_reused_and_partial_file_ignored(self):
        import numpy as np
        script = Path(__file__).resolve().parents[1]/'scripts/run_phase2_controls.py'
        spec = importlib.util.spec_from_file_location('phase2_controls_test',script)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        calls = []
        class Encoder:
            def __init__(self,*args,**kwargs):
                self.metadata = {'repo':'swiss-ai/Apertus-v1.5-8B','precision':'FAKE TEST ONLY'}
            def infer(self,messages,**kwargs):
                calls.append(messages)
                return {'hidden':np.zeros(4096),'option_logits':np.zeros(3),'label':0,
                        'probabilities':[.8,.1,.1],'context_tokens':100,'latency_seconds':.01}
        module = types.ModuleType('ost_nli.v15')
        module.Engine = Encoder
        module.REVISION = 'a411d838600baf0e3635a3daf66fb7c55fc97bb6'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source,out = root/'inputs',root/'out'
            source.mkdir()
            manifest = {'model_revision':module.REVISION,'conditions':{}}
            for split in ('train','validation'):
                rows = [{'id':f'{split}-{i}','label':i,'claim_language':'de','document_language':'de',
                         'messages':[{'role':'user','content':'claim only'}], 'context_bytes':0,
                         'retrieval_seconds':0,'truncated':False,'evidence_ids':[],'evidence_pages':[]} for i in range(3)]
                file = source/f'{split}-claim-only.jsonl'
                file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                manifest['conditions'][f'{split}-claim-only'] = {'file':file.name,'sha256':runner.sha(file),
                                                                 'dataset_sha256':'fixture'}
            (source/'manifest.json').write_text(json.dumps(manifest))
            argv = ['run_phase2_controls.py','--model-dir','unused','--inputs',str(source),
                    '--output',str(out),'--conditions','claim-only']
            with patch.dict(sys.modules,{'ost_nli.v15':module}),patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                self.assertEqual(len(calls),6)
                checkpoint = out/'claim-only/validation-chunks/000000.npz'
                preserved = checkpoint.read_bytes()
                checkpoint.with_suffix('.partial.npz').write_bytes(b'incomplete write')
                runner.main()
                self.assertEqual(len(calls),6,'Completed model calls must not be repeated')
                self.assertEqual(checkpoint.read_bytes(),preserved)
                manifest['extra'] = 'changed input contract'
                (source/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'contract changed'):
                    runner.main()


if __name__ == '__main__':
    unittest.main()
