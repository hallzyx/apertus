"""Evaluate a previously frozen decision architecture once on internal test data."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from ost_nli.data import fingerprint,load_dataset,words
from ost_nli.experiments import append_event,utc_now
from ost_nli.frozen import FrozenApertus,SYSTEM
from ost_nli.metrics import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--head-dir',required=True)
    p.add_argument('--splits',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=p.parse_args();out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Final evaluation output cannot be overwritten')
    splits=Path(args.splits);rows={name:load_dataset(splits/f'{name}.jsonl') for name in ['train','validation','test']}
    saved=json.loads((Path(__file__).resolve().parents[1]/'experiments/split_manifest.json').read_text())
    for name,data in rows.items():
        if fingerprint(data)!=saved['partitions'][name]['sha256']:raise ValueError('Frozen partition fingerprint mismatch')
    for name in ['train','validation']:
        if {r['booklet_id'] for r in rows[name]}&{r['booklet_id'] for r in rows['test']}:raise ValueError('Test booklet leakage')
        if {' '.join(words(r['claim'])) for r in rows[name]}&{' '.join(words(r['claim'])) for r in rows['test']}:raise ValueError('Test claim leakage')
    headroot=Path(args.head_dir);head=json.loads((headroot/'experiment.json').read_text())
    expected_train=head.get('train_sha256',head.get('training_dataset_sha256'))
    if expected_train!=fingerprint(rows['train']):raise ValueError('Head training fingerprint mismatch')
    out.mkdir(parents=True)
    launch={'experiment_id':out.name,'status':'started','timestamp':utc_now(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'head_experiment':head['experiment_id'],'head_parameters_sha256':hashlib.sha256((headroot/'head.json').read_bytes()).hexdigest(),
        'selection':'Frozen validation-selected architecture; no test fitting, remapping, calibration or threshold changes',
        'dataset_split':'strict_internal_test','test_sha256':fingerprint(rows['test']),
        'model':head['model'],'model_revision':head['model_revision'],'prompt_sha256':hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'macro_f1':None,'official_challenge_score':False,'actual_compute_cost':None,
        'notes':'Provided references, no gold or full-booklet accuracy. Internal test only; challenge rules still unverified.'}
    (out/'experiment.json').write_text(json.dumps(launch,indent=2)+'\n')
    start=time.perf_counter()
    try:
        model=FrozenApertus(args.model_dir,args.head_dir,args.device)
        predictions=[]
        for index,row in enumerate(rows['test']):
            actual=model.predict({'document_id':row['booklet_id'],'passages':row['passages']},row['claim'])
            pred={**actual,'id':row['id']}
            predictions.append(pred);append_event(out/'predictions.jsonl',pred)
            if (index+1)%25==0 or index==len(rows['test'])-1:print('Internal final test',index+1,'/',len(rows['test']),flush=True)
        metrics=evaluate(rows['test'],predictions)
        final={**launch,'status':'completed','timestamp':utc_now(),'n':len(rows['test']),
            'runtime_seconds':time.perf_counter()-start,'gpu':model.torch.cuda.get_device_name(0) if args.device=='cuda' else 'CPU',
            'macro_f1':metrics['macro_f1'],'average_context_tokens':metrics['average_context_tokens'],
            'average_inference_latency':metrics['average_latency_seconds'],'calibration_metrics':metrics['calibration'],
            'per_class_f1':{k:v['f1'] for k,v in metrics['per_class'].items()},
            'per_language_performance':metrics['by_language'],'cross_lingual_performance':metrics['cross_lingual']}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        (out/'predictions.json').write_text(json.dumps({'dataset_sha256':fingerprint(rows['test']),'configuration':final,'predictions':predictions},ensure_ascii=False,indent=2)+'\n')
        final['files']=[{'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in sorted(out.iterdir()) if path.name!='experiment.json']
        (out/'experiment.json').write_text(json.dumps(final,indent=2)+'\n');print('Internal test Macro-F1',metrics['macro_f1'],flush=True)
    except Exception as error:
        (out/'experiment.json').write_text(json.dumps({**launch,'status':'failed','timestamp':utc_now(),'error_type':type(error).__name__},indent=2)+'\n');raise


if __name__=='__main__':main()
