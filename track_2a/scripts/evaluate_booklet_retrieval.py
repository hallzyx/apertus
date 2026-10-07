"""Reference-overlap diagnostics for real full-booklet retrieval, not NLI F1."""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path


def main():
    from ost_nli.data import load_dataset, fingerprint
    from ost_nli.dense import MultilingualRetriever
    from ost_nli.documents import reference_coverage
    from ost_nli.retrieval import retrieve
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full', required=True); p.add_argument('--reference', required=True)
    p.add_argument('--embedding-dir', required=True); p.add_argument('--cache-dir', required=True)
    p.add_argument('--output', required=True); p.add_argument('--k', type=int, default=5)
    a = p.parse_args(); full = load_dataset(a.full); refs = load_dataset(a.reference)
    if [r['id'] for r in full] != [r['id'] for r in refs]: raise ValueError('Complete identical example IDs required')
    out = Path(a.output)
    if out.exists(): raise ValueError('Existing retrieval experiment cannot be overwritten')
    out.mkdir(parents=True)
    dense = MultilingualRetriever(a.embedding_dir, a.cache_dir)
    records = []
    for row, ref in zip(full, refs):
        reference = '\n\n'.join(p['text'] for p in ref['passages'])
        for mode in ['bm25', 'dense', 'hybrid']:
            began = time.perf_counter()
            selected = retrieve(row['passages'], row['claim'], a.k) if mode == 'bm25' else dense.retrieve(row['passages'], row['claim'], a.k, mode)
            record = {'id': row['id'], 'mode': mode, 'claim_language': row['claim_language'],
                'document_language': row['document_language'], 'latency_seconds': time.perf_counter()-began,
                'reference_5gram_coverage': reference_coverage(selected, reference),
                'full_document_reference_5gram_coverage': reference_coverage(row['passages'], reference),
                'selected_characters': sum(len(p['text']) for p in selected),
                'passage_ids': [p['id'] for p in selected], 'pages': [p['page'] for p in selected]}
            records.append(record)
        if len(records) % 75 == 0: print('Retrieved real booklets:', len(records)//3, '/',len(full), flush=True)
    def summarize(values):
        valid = [r for r in values if r['reference_5gram_coverage'] is not None]
        return {'n':len(values), 'reference_5gram_mean_coverage':sum(r['reference_5gram_coverage'] for r in valid)/len(valid),
            'fraction_with_at_least_90_percent_reference_coverage':sum(r['reference_5gram_coverage']>=.9 for r in valid)/len(valid),
            'average_selected_characters':sum(r['selected_characters'] for r in values)/len(values),
            'average_retrieval_seconds_including_first_document_encoding':sum(r['latency_seconds'] for r in values)/len(values)}
    results = {}
    for mode in ['bm25','dense','hybrid']:
        values = [r for r in records if r['mode']==mode]
        results[mode] = {**summarize(values), 'by_language_pair': {
            pair:summarize([r for r in values if r['document_language']+'->'+r['claim_language']==pair])
            for pair in sorted({r['document_language']+'->'+r['claim_language'] for r in values})},
            'cross_lingual':summarize([r for r in values if r['document_language']!=r['claim_language']])}
    result = {'full_sha256':fingerprint(full),'reference_sha256':fingerprint(refs),'n':len(full),'k':a.k,
        'model':'intfloat/multilingual-e5-small','results':results,'macro_f1':None,
        'notes':'Real PDFs and embeddings. Reference-overlap diagnostic, not official evidence scoring or NLI accuracy. Source references are never retrieval inputs. No parameter fitting or test-label selection.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(json.dumps({k:v['reference_5gram_mean_coverage'] for k,v in results.items()}),flush=True)


if __name__ == '__main__': main()
