"""Task lease bootstrap with ephemeral RSA-encrypted model credential delivery.

The caller controls the task-owned Vast onstart command. No plaintext credential
is written to disk, onstart configuration, logs or repository. A temporary private
transport key is deleted before model download; the credential exists in RAM only.
"""
import argparse
import base64
import json
import os
import subprocess
from pathlib import Path


def run(args):subprocess.run(args,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sealed-token')
    a=p.parse_args()
    root=Path('/workspace/apertus');os.chdir(root)
    os.environ.update(PYTHONPATH='track_2a/src',HF_HUB_DISABLE_XET='1',HF_HUB_DISABLE_TELEMETRY='1',
        HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HOME='/workspace/hf',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    private=Path('/workspace/v15-hf-transport.pem')
    if not a.sealed_token:
        if not private.exists():
            old=os.umask(0o077)
            try:run(['openssl','genpkey','-algorithm','RSA','-pkeyopt','rsa_keygen_bits:3072','-out',str(private)])
            finally:os.umask(old)
        public=subprocess.check_output(['openssl','pkey','-in',str(private),'-pubout'])
        print('HF_TRANSPORT_PUBLIC_KEY='+base64.b64encode(public).decode(),flush=True)
        return
    if not Path('/workspace/model/verified_manifest.json').exists():
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
    if not reference.exists():run(['python','-m','ost_nli','prepare-ost','--output',str(reference)])
    if not (splits/'manifest.json').exists():run(['python','-m','ost_nli','split',str(reference),'--output-dir',str(splits),'--seed','42'])
    expected=json.loads(Path('track_2a/experiments/split_manifest.json').read_text())
    if json.loads((splits/'manifest.json').read_text())!=expected:raise ValueError('Frozen split mismatch')
    full=Path('track_2a/data/private/full-booklets-v2')
    if not (full/'audit.json').exists():
        run(['python','track_2a/scripts/prepare_booklets.py','--source','track_2a/data/private/official/v1.1.jsonl',
            '--splits',str(splits),'--output',str(full)])
    audit=json.loads((full/'audit.json').read_text())
    frozen=json.loads(Path('track_2a/experiments/booklet-source-v2/audit.json').read_text())
    for split in ['train','validation','test']:
        if audit['partitions'][split]['sha256']!=frozen['partitions'][split]['sha256']:raise ValueError('Full booklet fingerprint mismatch')
    from ost_nli.dense import download as download_e5
    if not Path('/workspace/e5/verified_manifest.json').exists():download_e5('/workspace/e5')
    if not Path('/workspace/v15-inputs/manifest.json').exists():
        run(['python','track_2a/scripts/prepare_v15_inputs.py','--full-dir',str(full),'--reference-dir',str(splits),
            '--embedding-dir','/workspace/e5','--cache-dir','/workspace/e5-vectors','--output','/workspace/v15-inputs'])
    print('V15_REPRODUCED_BOOKLET_INPUTS_VERIFIED',flush=True)
    run(['bash','track_2a/scripts/vast_v15_worker.sh'])


if __name__=='__main__':main()
