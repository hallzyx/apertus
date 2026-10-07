"""Registered exact prompt-token context packing; no reference passed to Apertus."""
import argparse
import hashlib
import json
import time
from pathlib import Path

CONDITIONS = {'hybrid-1k':(1024,5),'hybrid-2k':(2048,10),
              'hybrid-4k':(4096,20),'hybrid-8k':(8192,30),'prefix-2k':(2048,None)}


def main():
    from transformers import AutoTokenizer
    from ost_nli.data import load_dataset, fingerprint, words
    from ost_nli.model import nli_messages, select_context
    from ost_nli.dense import MultilingualRetriever, REVISION as E5_REVISION
    from ost_nli.v15 import REVISION
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--reference-data',required=True)
    p.add_argument('--model-dir',required=True);p.add_argument('--retriever-dir',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Frozen context inputs already exist')
    out.mkdir(parents=True)
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True,trust_remote_code=False)
    retriever=MultilingualRetriever(a.retriever_dir)
    def token_count(context,claim):
        text=tokenizer.apply_chat_template(nli_messages(context,claim),tokenize=False,
             add_generation_prompt=True,enable_thinking=False)+'{"label":'
        return len(tokenizer(text,add_special_tokens=False)['input_ids'])
    manifest={'study':'registered_context_sweep','scope':'train902/validation276 only; no consumed310',
              'model_revision':REVISION,'embedding_revision':E5_REVISION,'conditions':{},
              'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'packing':'Whole exact passages; hybrid skips oversize candidates, prefix stops; total prompt cap includes system, claim and assistant prefix',
              'caution':'Depth and token cap varied jointly under bounded budget; not a factorial attribution of their effects'}
    for split in ('train','validation'):
        rows=load_dataset(Path(a.data)/f'{split}.jsonl')
        references={r['id']:r for r in load_dataset(Path(a.reference_data)/f'{split}.jsonl')}
        records={c:[] for c in CONDITIONS}
        for index,row in enumerate(rows):
            began=time.perf_counter();ranked=retriever.retrieve(row['passages'],row['claim'],k=30,mode='hybrid')
            retrieval_seconds=time.perf_counter()-began
            for condition,(cap,depth) in CONDITIONS.items():
                began=time.perf_counter();candidates=ranked[:depth] if depth else row['passages']
                selected=[];context=''
                if token_count('',row['claim'])>cap:raise ValueError('Claim exceeds registered prompt budget')
                for passage in candidates:
                    trial,trial_context=select_context({'passages':selected+[passage]},mode='full',max_bytes=1000000)
                    if token_count(trial_context,row['claim'])>cap:
                        if depth is None:break
                        continue
                    selected,context=trial,trial_context
                used=token_count(context,row['claim'])
                record={k:row[k] for k in ('id','label','booklet_id','claim','claim_language','document_language','document_id')}
                def grams(text):
                    w=words(text);return set(tuple(w[i:i+5]) for i in range(max(0,len(w)-4)))
                reference=grams(' '.join(p['text'] for p in references[row['id']]['passages']))
                supplied=grams(' '.join(p['text'] for p in selected))
                record.update(messages=nli_messages(context,row['claim']),context_bytes=len(context.encode()),
                    evidence_ids=[p['id'] for p in selected],evidence_pages=[p.get('page') for p in selected],
                    retrieval_seconds=(retrieval_seconds if depth else 0)+time.perf_counter()-began,
                    condition=condition,truncated=len(selected)<len(row['passages']),prompt_token_cap=cap,
                    prepared_prompt_tokens=used,retrieval_depth=depth,
                    development_reference_5gram_coverage=len(reference&supplied)/len(reference) if reference else None,
                    selected_passages=selected)
                records[condition].append(record)
            if (index+1)%50==0 or index+1==len(rows):print('CONTEXT_PREPARE',split,index+1,len(rows),flush=True)
        for condition,values in records.items():
            path=out/f'{split}-{condition}.jsonl'
            path.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
            manifest['conditions'][f'{split}-{condition}']={'file':path.name,'n':len(values),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dataset_sha256':fingerprint(rows),
                'prompt_cap':CONDITIONS[condition][0],'retrieval_depth':CONDITIONS[condition][1],
                'mean_prepared_prompt_tokens':sum(v['prepared_prompt_tokens'] for v in values)/len(values),
                'mean_reference_5gram_coverage':sum(v['development_reference_5gram_coverage'] or 0 for v in values)/len(values),
                'reference_overlap_caution':'Development lexical diagnostic only; Neutral reference may deliberately be unrelated; not organizer evidence scoring'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('CONTEXT_INPUTS_PREPARED',flush=True)


if __name__=='__main__':main()
