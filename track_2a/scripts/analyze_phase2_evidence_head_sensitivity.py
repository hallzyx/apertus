"""CPU-only replay of already-frozen heads on actual evidence-only native features."""
import hashlib,json
from pathlib import Path

def main():
 import numpy as np
 phase=Path('track_2a/experiments/apertus-v15-phase2');out=phase/'evidence-head-sensitivity.json'
 if out.exists():raise ValueError('Preserve immutable diagnostic')
 sources=[]
 for name in ('final','accuracy'):
  root=Path(f'track_2a/experiments/apertus-v15-phase2-{name}-validation-v1')
  if not root.exists():continue
  metadata=json.loads((root/'experiment.json').read_text());assert metadata['status']=='completed'
  feature_path=root/'bounded-diagnostic-features.npz';entry=next(v for v in metadata['files'] if v['path']==feature_path.name);assert hashlib.sha256(feature_path.read_bytes()).hexdigest()==entry['sha256']
  diagnostics=json.loads((root/'bounded-context-evidence-diagnostics.json').read_text());proofs={v['id']:v for v in diagnostics['evidence_proofs']}
  with np.load(feature_path,allow_pickle=False) as cache:
   mask=cache['modes']=='evidence-only';ids=cache['ids'][mask].tolist();x=cache['hidden'][mask].astype(np.float64);assert set(ids)==set(proofs)
   logits=cache['option_logits'][mask].astype(np.float64)
  heads={}
  for context,cap in (('hybrid-1k',1024),('hybrid-2k',2048),('hybrid-4k',4096),('hybrid-8k',8192)):
   path=Path(f'track_2a/experiments/apertus-v15-phase2-context-v1/{context}-fitted-head/head.json');h=json.loads(path.read_text())
   z=((x-np.asarray(h['feature_mean']))/np.asarray(h['feature_scale']))@np.asarray(h['coefficients']).T+np.asarray(h['intercept']);z/=h['temperature'];q=np.exp(z-z.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
   values=[{'id':iid,'original_long_context_decision':proofs[iid]['original_label'],'replayed_short_evidence_decision':int(p.argmax()),
    'probabilities':p.tolist(),'same_decision_head':context==metadata['context'],'within_head_token_cap':proofs[iid]['output']['context_tokens']<=cap} for iid,p in zip(ids,q)]
   heads[context]={'head_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'decision_preservation_rate':sum(v['original_long_context_decision']==v['replayed_short_evidence_decision'] for v in values)/len(values),'n':len(values),'rows':values}
   if context==metadata['context']:
    assert abs(heads[context]['decision_preservation_rate']-diagnostics['evidence_decision_preservation_rate'])<1e-12
    assert all(np.allclose(p,proofs[iid]['output']['probabilities'],atol=1e-6) for iid,p in zip(ids,q))
  sources.append({'experiment':str(root),'original_context':metadata['context'],'source_feature_sha256':entry['sha256'],
   'sample_caution':'Separate predetermined predicted-class/language samples per candidate; rates across4k/8k are not necessarily paired because case IDs can differ.',
   'frozen_head_replays':heads,'native_restricted_score_decision_preservation':sum(int(p.argmax())==proofs[iid]['original_label'] for iid,p in zip(ids,logits))/len(ids)})
 result={'scope':'Cached actual Apertus evidence-only features, unchanged previously fitted heads. No new model call/head fit/temperature or hidden310 access.',
  'interpretation':'Alternative short-context heads cannot certify semantic relevance or prove causality. This tests sensitivity to a different frozen decision mapping; it is not automatic evidence rejection or a production policy. Low preservation remains an important limitation.',
  'heads_adopted':False,'experiments':sources};out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('Frozen evidence-head sensitivity recorded without new model calls')
if __name__=='__main__':main()
