"""Select train/validation by original line ID before parsing labels; no holdout rows."""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def main():
    from ost_nli.official import URL, SOURCE_SHA256, REVISION
    from ost_nli.data import fingerprint, load_dataset
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='track_2a/data/private/official/v1.1.jsonl')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    source,out = Path(a.source),Path(a.output)
    if out.exists():
        raise ValueError('Immutable phase2 source output exists')
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(URL,timeout=90) as response:
            data = response.read(30_000_001)
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise ValueError('Official source checksum mismatch')
        source.write_bytes(data)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned official source bytes required')
    manifest = json.loads((Path(__file__).resolve().parents[1]/'experiments/split_manifest.json').read_text())
    selected = {split:set(manifest['partitions'][split]['example_ids']) for split in ('train','validation')}
    allowed = selected['train'] | selected['validation']
    rows,raw_rows = [],[]
    with source.open() as stream:
        for index,line in enumerate(stream):
            identity = f'ost-v1.1-{index:04d}'
            if identity not in allowed:
                continue  # Do not decode, inspect or convert consumed holdout labels/claims.
            r = json.loads(line)
            raw_rows.append(r)
            ref,url,date = r['reference_string'],r['booklet_url'],r['booklet_publish_date']
            digest = hashlib.sha256(ref.encode()).hexdigest()
            parts = [s.strip() for s in re.split(r'\n\s*\n',ref) if s.strip()]
            passages = [{'id':f'{digest[:20]}-p{j+1}','page':None,'text':text,
                         'language':r['reference_language'],'source_url':url,'source_reference_sha256':digest} for j,text in enumerate(parts)]
            rows.append({'id':identity,'document_id':f'{date}-{r["reference_language"]}-{digest[:20]}',
                         'booklet_id':date,'claim':r['claim'],'claim_language':r['claim_language'],
                         'document_language':r['reference_language'],'label':r['entailment_label'],
                         'passages':passages,'evidence_ids':[],'evidence_annotation_available':False,
                         'provided_reference_ids':[p['id'] for p in passages],'document_scope':'provided_reference',
                         'source_booklet_url':url,'source_row_index':index,'source_revision':REVISION})
    if {r['id'] for r in rows} != allowed:
        raise ValueError('Missing registered train/validation IDs')
    out.mkdir(parents=True)
    (out/'selected-official.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in raw_rows))
    audit = {'scope':'Only original train/validation IDs decoded; source archive checksum covers complete downloaded file',
             'source_sha256':SOURCE_SHA256,'partitions':{}}
    for split in ('train','validation'):
        data = [r for r in rows if r['id'] in selected[split]]
        path = out/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
        verified = load_dataset(path)
        if fingerprint(verified) != manifest['partitions'][split]['sha256']:
            raise ValueError('Frozen reference split changed')
        audit['partitions'][split] = {'n':len(verified),'sha256':fingerprint(verified)}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('PHASE2_TRAIN_VALIDATION_SOURCE_VERIFIED',flush=True)


if __name__ == '__main__':
    main()
