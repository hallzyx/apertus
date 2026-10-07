"""Prepare deterministic grounding controls using train/validation only."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def main():
    from ost_nli.data import load_dataset, fingerprint
    from ost_nli.model import select_context, nli_messages
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='track_2a/data/private/full-booklets-v2')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError('Do not overwrite frozen phase2 inputs')
    out.mkdir(parents=True)
    manifest = {'scope':'train/validation only; no test access', 'conditions':{},
                'wrong_document_scoring':'Original labels only diagnose artifact retention; not true swapped-pair NLI labels',
                'model_revision':'a411d838600baf0e3635a3daf66fb7c55fc97bb6',
                'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for split in ('train','validation'):
        rows = load_dataset(Path(a.data)/f'{split}.jsonl')
        documents = {r['document_id']:r for r in rows}
        conditions = ['claim-only'] if split == 'train' else ['claim-only','generic','wrong-42','wrong-1337']
        for condition in conditions:
            target = out/f'{split}-{condition}.jsonl'
            records = []
            for row in rows:
                began = time.perf_counter()
                selected = []
                context = ''
                donor = None
                if condition == 'generic':
                    context = {'de':'Dieses Dokument enthält allgemeine Informationen zu einer eidgenössischen Abstimmung.',
                               'fr':'Ce document contient des informations générales sur une votation fédérale.',
                               'it':'Questo documento contiene informazioni generali su una votazione federale.'}[row['document_language']]
                elif condition.startswith('wrong-'):
                    candidates = sorted((r for r in documents.values() if r['booklet_id']!=row['booklet_id']
                                         and r['document_language']==row['document_language']),key=lambda r:r['document_id'])
                    if not candidates:
                        raise ValueError('No different-event same-language donor')
                    seed = int(condition.split('-')[1])
                    donor = random.Random(f'{seed}:{row["id"]}').choice(candidates)
                    selected,context = select_context(donor,mode='full')
                    assert donor['document_id'] != row['document_id'] and donor['booklet_id'] != row['booklet_id']
                record = {k:row[k] for k in ('id','label','booklet_id','claim','claim_language','document_language','document_id')}
                record.update(messages=nli_messages(context,row['claim']),context_bytes=len(context.encode()),
                              evidence_ids=[r['id'] for r in selected],evidence_pages=[r.get('page') for r in selected],
                              retrieval_seconds=time.perf_counter()-began,condition=condition,
                              donor_document_id=donor['document_id'] if donor else None,
                              donor_booklet_id=donor['booklet_id'] if donor else None,
                              donor_language=donor['document_language'] if donor else None,
                              truncated=bool(donor and (len(selected)<len(donor['passages']) or any(r['truncated'] for r in selected))))
                records.append(record)
            target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
            manifest['conditions'][f'{split}-{condition}'] = {'file':target.name,'n':len(rows),
                'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'dataset_sha256':fingerprint(rows),
                'mean_context_bytes':sum(r['context_bytes'] for r in records)/len(records),
                'same_language_donor_rate':1. if condition.startswith('wrong-') else None,
                'different_event_donor_rate':1. if condition.startswith('wrong-') else None}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
