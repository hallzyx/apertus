"""Supplementary E5 reference alignment on validation only; no official evidence scoring."""
import argparse,hashlib,json
from pathlib import Path

def main():
 import numpy as np
 from ost_nli.data import load_dataset,fingerprint
 from ost_nli.dense import MultilingualRetriever,REVISION
 from ost_nli.model import select_context
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',required=True);p.add_argument('--references',required=True);p.add_argument('--cache-dir',required=True)
 a=p.parse_args();root=Path('track_2a');phase=root/'experiments/apertus-v15-phase2';out=phase/'semantic-reference-diagnostic.json'
 if out.exists():raise ValueError('Preserve immutable diagnostic')
 rows=load_dataset(root/'data/private/full-booklets-v2/validation.jsonl');assert fingerprint(rows)=='00839ac3040e4a75cc4a122338a62a38919bb997648aeb3a9af7f7415d93e74f'
 references=load_dataset(a.references);refs={r['id']:r for r in references};assert set(refs)=={r['id'] for r in rows}
 retriever=MultilingualRetriever(a.model_dir,a.cache_dir)
 texts=['\n\n'.join(p['text'] for p in refs[r['id']]['passages']) for r in rows]
 queries=retriever.encode(texts,'query: ');selected={'original-full':{r['id']:{p['id'] for p in select_context(r,mode='full')[0]} for r in rows}}
 source_paths=[]
 for condition in ('hybrid-1k','hybrid-2k','hybrid-4k','hybrid-8k','prefix-2k'):
  path=root/'experiments/apertus-v15-phase2-context-v1'/condition/'validation-evidence.jsonl';records=[json.loads(s) for s in path.read_text().splitlines()];assert {v['id'] for v in records}==set(refs)
  selected[condition]={v['id']:{p['id'] for p in v['selected_passages']} for v in records}
  source_paths.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
 values=[]
 for index,row in enumerate(rows):
  passages=row['passages'];key=hashlib.sha256(json.dumps([(p['id'],p['text']) for p in passages],ensure_ascii=False).encode()).hexdigest()
  if key not in retriever.documents:retriever.retrieve(passages,row['claim'],k=1,mode='dense')
  scores=retriever.documents[key]@queries[index];order=sorted(range(len(passages)),key=lambda i:(-float(scores[i]),i));top=[passages[i]['id'] for i in order[:3]]
  values.append({'id':row['id'],'gold':row['label'],'claim_language':row['claim_language'],'document_language':row['document_language'],
   'reference_query_truncated_at512':len(retriever.tokenizer.encode('query: '+texts[index]))>512,
   'best_same_booklet_reference_similarity':float(scores[order[0]]),'reference_best_match_ids':top,
   'conditions':{name:{'top1_presence':top[0] in ids[row['id']], 'top3_presence_fraction':len(set(top)&ids[row['id']])/len(top),
    'best_selected_reference_similarity':max((float(scores[i]) for i,p in enumerate(passages) if p['id'] in ids[row['id']]),default=None)} for name,ids in selected.items()}})
  if (index+1)%30==0:print('SEMANTIC_REFERENCE_PROGRESS',index+1,len(rows),flush=True)
 def summarize(subset):
  return {'n':len(subset),'conditions':{name:{'top1_presence_rate':sum(v['conditions'][name]['top1_presence'] for v in subset)/len(subset),
   'top3_presence_fraction':float(np.mean([v['conditions'][name]['top3_presence_fraction'] for v in subset])),
   'mean_best_selected_reference_similarity':float(np.mean([v['conditions'][name]['best_selected_reference_similarity'] for v in subset if v['conditions'][name]['best_selected_reference_similarity'] is not None]))} for name in selected}}
 result={'scope':'Supplementary validation-only semantic reference alignment using the same frozen E5 supporting retriever, NOT independent evidence judging, true evidence recall or organizer scoring.',
  'definition':'Rank all same-booklet passages by E5 cosine with the development reference as query; measure whether the reference-nearest1/3 passages occur in actual model input. Best matches are pseudo-targets, not gold relevance labels.',
  'limitations':'Same encoder used for retrieval gives dependent diagnostic. References and passages truncated to512 E5 tokens. Neutral references may be unrelated; report them descriptively, never interpret missing reference as Neutral retrieval failure. References never enter production or Apertus messages.',
  'embedding_revision':REVISION,'validation_sha256':fingerprint(rows),'references_sha256':fingerprint(references),'source_evidence':source_paths,
  'overall':summarize(values),'entailment_contradiction':summarize([v for v in values if v['gold']!=1]),'neutral_descriptive_only':summarize([v for v in values if v['gold']==1]),
  'by_claim_language':{lang:summarize([v for v in values if v['claim_language']==lang]) for lang in ('de','fr','it')},
  'cross_lingual':summarize([v for v in values if v['claim_language']!=v['document_language']]),'rows':values,'training_performed':False,'consumed310_accessed':False}
 out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SEMANTIC_REFERENCE_COMPLETED',flush=True)
if __name__=='__main__':main()
