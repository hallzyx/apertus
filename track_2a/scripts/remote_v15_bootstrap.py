"""Task lease bootstrap with ephemeral RSA-encrypted model credential delivery.

The caller controls the task-owned Vast onstart command. No plaintext credential
is written to disk, onstart configuration, logs or repository. A temporary private
transport key is deleted before model download; the credential exists in RAM only.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path


def run(args):subprocess.run(args,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sealed-token')
    p.add_argument('--token-env', help='Provider-managed temporary read credential; removed from child process environment')
    p.add_argument('--allow-missing-training-source',action='store_true',help='Declare a training subset selected only by PDF availability; validation/test must stay complete')
    a=p.parse_args()
    root=Path('/workspace/apertus');os.chdir(root)
    os.environ.update(PYTHONPATH='track_2a/src',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HOME='/workspace/hf',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    private=Path('/workspace/v15-hf-transport.pem')
    if not a.sealed_token and not a.token_env:
        if not private.exists():
            old=os.umask(0o077)
            try:run(['openssl','genpkey','-algorithm','RSA','-pkeyopt','rsa_keygen_bits:3072','-out',str(private)])
            finally:os.umask(old)
        public=subprocess.check_output(['openssl','pkey','-in',str(private),'-pubout'])
        Path('/workspace/v15-transport-public.pem').write_bytes(public)
        encoded=base64.b64encode(public).decode()
        for i in range(0,len(encoded),400):
            print('HF_TRANSPORT_PUBLIC_PART='+str(i//400)+':'+encoded[i:i+400],flush=True)
        print('HF_TRANSPORT_PUBLIC_SHA256='+hashlib.sha256(public).hexdigest(),flush=True)
        return
    if not Path('/workspace/model/verified_manifest.json').exists():
        if a.token_env:
            token=os.environ.pop(a.token_env, '')
            if not token:raise ValueError('Provider-managed read credential was not injected')
        else:
            encrypted=base64.b64decode(a.sealed_token,validate=True)
            token=subprocess.check_output(['openssl','pkeyutl','-decrypt','-inkey',str(private),
                '-pkeyopt','rsa_padding_mode:oaep','-pkeyopt','rsa_oaep_md:sha256'],input=encrypted).decode().strip()
            private.unlink()
        from ost_nli.v15 import download
        try:manifest=download('/workspace/model',token)
        finally:del token
        print('V15_WEIGHTS_VERIFIED '+manifest['revision'],flush=True)
    run(['python','-m','pip','install','--no-cache-dir','--require-hashes','-r','track_2a/requirements-pdf.lock'])
    reference=Path('track_2a/data/private/ost-reference.jsonl')
    splits=Path('track_2a/data/private/splits-strict')
    if not reference.exists():run(['python','-m','ost_nli','prepare-ost','--source','track_2a/data/private/official/v1.1.jsonl','--output',str(reference)])
    if not (splits/'manifest.json').exists():run(['python','-m','ost_nli','split',str(reference),'--output-dir',str(splits),'--seed','42'])
    expected=json.loads(Path('track_2a/experiments/split_manifest.json').read_text())
    if json.loads((splits/'manifest.json').read_text())!=expected:raise ValueError('Frozen split mismatch')
    full=Path('track_2a/data/private/full-booklets-v2')
    existing=json.loads((full/'audit.json').read_text()) if (full/'audit.json').exists() else None
    if not existing or existing.get('booklets_downloaded')!=existing.get('booklets_total'):
        run(['python','track_2a/scripts/prepare_booklets.py','--source','track_2a/data/private/official/v1.1.jsonl',
            '--splits',str(splits),'--output',str(full)])
    audit=json.loads((full/'audit.json').read_text())
    frozen=json.loads(Path('track_2a/experiments/booklet-source-v2/audit.json').read_text())
    for split in ['validation','test']:
        if audit['partitions'][split]['sha256']!=frozen['partitions'][split]['sha256']:raise ValueError('Full booklet fingerprint mismatch')
    partial_train=audit['partitions']['train']['sha256']!=frozen['partitions']['train']['sha256']
    if partial_train:
        if not a.allow_missing_training_source:raise ValueError('Full training booklet fingerprint mismatch')
        from ost_nli.data import load_dataset
        original={r['id']:r for r in load_dataset(splits/'train.jsonl')}
        available=load_dataset(full/'train.jsonl');ids={r['id'] for r in available}
        if len(available)<500 or len(ids)!=len(available) or not ids<=original.keys():raise ValueError('Invalid training availability subset')
        if set(original)-ids!=set(audit['partitions']['train']['missing_ids']):raise ValueError('Unexplained training exclusions')
        for row in available:
            if any(row[k]!=original[row['id']][k] for k in ['label','claim','booklet_id','claim_language','document_language','source_booklet_url']):
                raise ValueError('Training subset changed canonical labels/claims/groups')
        print(f'V15_DECLARED_TRAIN_SUBSET {len(available)}/{len(original)}; validation/test complete; exclusions depend only on unavailable source PDFs',flush=True)
    from ost_nli.dense import download as download_e5
    if not Path('/workspace/e5/verified_manifest.json').exists():download_e5('/workspace/e5')
    if not Path('/workspace/v15-inputs/manifest.json').exists():
        command=['python','track_2a/scripts/prepare_v15_inputs.py','--full-dir',str(full),'--reference-dir',str(splits),
            '--embedding-dir','/workspace/e5','--cache-dir','/workspace/e5-vectors','--output','/workspace/v15-inputs']
        if partial_train:command.append('--allow-train-subset')
        run(command)
    print('V15_REPRODUCED_BOOKLET_INPUTS_VERIFIED',flush=True)
    run(['bash','track_2a/scripts/vast_v15_worker.sh'])
    run(['python','track_2a/scripts/finish_v15_research.py'])


if __name__=='__main__':main()
