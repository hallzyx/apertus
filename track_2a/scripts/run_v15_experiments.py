"""Real v1.5 validation ablations and train-only feature extraction, no test."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def save(path,obj): Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def main():
    import numpy as np
    from ost_nli.v15 import Engine
    from ost_nli.metrics import evaluate
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=['validation','train'],required=True)
    p.add_argument('--context',default='hybrid')
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Existing experiment output cannot be overwritten')
    out.mkdir(parents=True)
    manifest=json.loads((Path(a.inputs)/'manifest.json').read_text())
    began=time.perf_counter();engine=Engine(a.model_dir)
    metadata={**engine.metadata,'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'source_engine_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'src/ost_nli/v15.py').read_bytes()).hexdigest(),
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_manifest_sha256':hashlib.sha256((Path(a.inputs)/'manifest.json').read_bytes()).hexdigest(),
        'load_seconds':time.perf_counter()-began,'status':'started','mode':a.mode,'training':'Frozen weights; no evaluation labels used in inference'}
    save(out/'experiment.json',metadata)
    modes=['reference','hybrid','dense','bm25','full'] if a.mode=='validation' else [a.context]
    for context in modes:
        source=Path(a.inputs)/f'{a.mode}-{context}.jsonl'
        entry=manifest['splits'][a.mode]['files'][context]
        if hashlib.sha256(source.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Frozen input checksum mismatch')
        rows=[json.loads(x) for x in source.read_text().splitlines()]
        methods=['score','prompt'] if a.mode=='validation' else ['score']
        for method in methods:
            directory=out/f'{context}-{method}';directory.mkdir()
            preds=[];hidden=[];logits=[]
            for i,row in enumerate(rows):
                result=engine.infer(row['messages'],method=method)
                if method=='score':hidden.append(result.pop('hidden'));logits.append(result.pop('option_logits'))
                else:result.pop('hidden');result.pop('option_logits')
                result.update(id=row['id'],retrieval_seconds=row['retrieval_seconds'],
                    model_inference_seconds=result['latency_seconds'],truncated=row['truncated'],
                    evidence_ids=row['evidence_ids'],evidence_pages=row['evidence_pages'])
                result['latency_seconds']+=row['retrieval_seconds'];preds.append(result)
                with (directory/'predictions.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
                if (i+1)%25==0:print(f'{a.mode} {context}/{method} {i+1}/{len(rows)} {result["model_inference_seconds"]:.3f}s {result["context_tokens"]} tokens',flush=True)
            if hidden:
                np.savez_compressed(directory/'features.npz',hidden=np.asarray(hidden,dtype=np.float32),
                    option_logits=np.asarray(logits,dtype=np.float32),ids=np.asarray([r['id'] for r in rows]),
                    labels=np.asarray([r['label'] for r in rows],dtype=np.int64))
            valid=[r for r,pred in zip(rows,preds) if pred['label'] is not None]
            valid_preds=[pred for pred in preds if pred['label'] is not None]
            metrics=evaluate([{**r,'evidence_ids':[]} for r in valid],valid_preds)
            metrics.update(n_total=len(rows),invalid_outputs=len(rows)-len(valid),valid_output_coverage=len(valid)/len(rows),
                scope='Validation internal benchmark; invalid prompt outputs excluded from subset score, counted separately',
                n_truncated=sum(p['truncated'] for p in preds))
            save(directory/'metrics.json',metrics)
            save(directory/'experiment.json',{**metadata,'status':'completed','context':context,'method':method,
                'dataset_sha256':manifest['splits'][a.mode]['full_sha256'],'macro_f1':metrics['macro_f1'],
                'n':len(rows),'invalid_outputs':metrics['invalid_outputs']})
            print(f'COMPLETE {a.mode} {context}/{method} macro_f1={metrics["macro_f1"]:.6f} invalid={metrics["invalid_outputs"]}',flush=True)
    save(out/'experiment.json',{**metadata,'status':'completed','runtime_seconds':time.perf_counter()-began})
    print('V15_EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':main()
