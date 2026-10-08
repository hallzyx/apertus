"""Fixed-grid cached4k→8k simulation; no router or model fitting."""
import hashlib,json
from pathlib import Path

def main():
 from ost_nli.data import load_dataset,fingerprint
 from ost_nli.metrics import evaluate
 root=Path('track_2a');phase=root/'experiments/apertus-v15-phase2';out=phase/'adaptive-8k-simulation.json'
 if out.exists():raise ValueError('Preserve immutable simulation')
 rows=load_dataset(root/'data/private/full-booklets-v2/validation.jsonl');assert fingerprint(rows)=='00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f'
 paths={name:root/'experiments/apertus-v15-phase2-context-v1'/f'{name}-fitted-head/predictions.json' for name in ('hybrid-4k','hybrid-8k')}
 sources={name:{r['id']:r for r in json.loads(path.read_text())['predictions']} for name,path in paths.items()};a,b=sources['hybrid-4k'],sources['hybrid-8k'];assert set(a)==set(b)=={r['id'] for r in rows}
 semantic_path=phase/'semantic-reference-diagnostic.json';semantic={r['id']:r for r in json.loads(semantic_path.read_text())['rows']};ec=[r for r in rows if r['label']!=1];simulations=[]
 for threshold in (.5,.7,.85,.95):
  predictions=[];escalated=[];routes={}
  for row in rows:
   first=a[row['id']];extra=max(first['probabilities'])<threshold;routes[row['id']]='hybrid-8k' if extra else 'hybrid-4k'
   result={**(b[row['id']] if extra else first)}
   if extra:
    result['context_tokens']+=first['context_tokens'];result['latency_seconds']+=first['latency_seconds'];escalated.append(row['id'])
   predictions.append(result)
  simulations.append({'threshold':threshold,'escalation_count':len(escalated),'escalation_rate':len(escalated)/len(rows),
   'mean_apertus_forwards':1+len(escalated)/len(rows),'metrics':evaluate(rows,predictions),
   'reference_nearest_quote_presence_EC':sum(semantic[r['id']]['conditions'][routes[r['id']]]['top1_presence'] for r in ec)/len(ec),
   'escalated_ids':escalated})
 result={'scope':'Additional CPU-only fixed-grid development simulation of4k→8k. No new Apertus forward, head fit, temperature fit or consumed310 access.',
  'threshold_grid':[.5,.7,.85,.95],'cost_policy':'When escalating, sum both actual cached encodings/context tokens plus separately measured retrieval/head time. Not live paired end-to-end latency.',
  'simulations':simulations,'source_predictions':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in paths.items()},
  'semantic_source_sha256':hashlib.sha256(semantic_path.read_bytes()).hexdigest(),'router_adopted':False,
  'decision':'The0.95 simulation is a useful lower-token trade-off, not a failed/dominated idea. Keep separately verified fixed8k for greater static Macro-F1 and reference-nearest source coverage, one head/one forward and simpler reproducibility. Live routing cost/integration and threshold transfer need independent-booklet validation. No new architecture chase or hidden-test tuning.'}
 out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps([{k:v[k] for k in ('threshold','escalation_rate','reference_nearest_quote_presence_EC')}|{k:v['metrics'][k] for k in ('macro_f1','average_context_tokens','average_latency_seconds')} for v in simulations],indent=2))
if __name__=='__main__':main()
