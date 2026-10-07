"""Freeze booklet-only contexts before GPU rental; references are oracle-only."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    from ost_nli.data import load_dataset, fingerprint
    from ost_nli.dense import MultilingualRetriever, REVISION
    from ost_nli.model import select_context, nli_messages
    from ost_nli.retrieval import retrieve
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-dir', required=True)
    p.add_argument('--reference-dir', required=True)
    p.add_argument('--embedding-dir', required=True)
    p.add_argument('--cache-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists(): raise ValueError('Frozen inputs must not be overwritten')
    out.mkdir(parents=True)
    dense = MultilingualRetriever(a.embedding_dir, a.cache_dir)
    manifest = {'k': 5, 'max_context_bytes': 48000, 'embedding_revision': REVISION,
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'production_input': 'full booklet + claim; references only in oracle experiments',
        'splits': {}}
    for split in ['validation', 'train', 'test']:
        rows = load_dataset(Path(a.full_dir)/f'{split}.jsonl')
        refs = load_dataset(Path(a.reference_dir)/f'{split}.jsonl')
        if [r['id'] for r in rows] != [r['id'] for r in refs]: raise ValueError('ID mismatch')
        modes = ['full', 'reference', 'bm25', 'dense', 'hybrid'] if split == 'validation' else ['bm25', 'dense', 'hybrid']
        paths = {mode: out/f'{split}-{mode}.jsonl' for mode in modes}
        handles = {mode: path.open('w') for mode, path in paths.items()}
        try:
            for i, (row, ref) in enumerate(zip(rows, refs)):
                for mode in modes:
                    began = time.perf_counter()
                    if mode in ['full', 'reference']:
                        selected, context = select_context(ref if mode == 'reference' else row, mode='full')
                    else:
                        candidates = retrieve(row['passages'],row['claim'],5) if mode == 'bm25' else dense.retrieve(row['passages'],row['claim'],5,mode)
                        selected, context = select_context({**row,'passages':candidates},mode='full')
                    record = {key:row[key] for key in ['id','label','booklet_id','claim','claim_language','document_language','document_id']}
                    record.update(messages=nli_messages(context,row['claim']), context_bytes=len(context.encode()),
                        evidence_ids=[p['id'] for p in selected], evidence_pages=[p.get('page') for p in selected],
                        retrieval_seconds=time.perf_counter()-began, mode=mode,
                        truncated=any(p['truncated'] for p in selected) or (mode=='full' and len(selected)<len(row['passages'])))
                    handles[mode].write(json.dumps(record,ensure_ascii=False)+'\n')
                if (i+1)%50 == 0: print(f'Frozen {split} contexts: {i+1}/{len(rows)}',flush=True)
        finally:
            for handle in handles.values(): handle.close()
        manifest['splits'][split] = {'n':len(rows),'full_sha256':fingerprint(rows),'reference_sha256':fingerprint(refs),
            'files':{mode:{'name':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for mode,path in paths.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Booklet contexts frozen; test labels never used for retrieval selection.',flush=True)


if __name__ == '__main__': main()
