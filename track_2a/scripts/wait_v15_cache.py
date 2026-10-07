"""Wait for a task-owned provider copy and verify the original model checksums."""
import argparse
import json
import time
from pathlib import Path


def main():
    from ost_nli.v15 import REPO,REVISION,digest
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True)
    parser.add_argument('--timeout',type=float,default=600)
    args=parser.parse_args();root=Path(args.directory);deadline=time.monotonic()+args.timeout
    print('V15_WAITING_FOR_PRIVATE_CACHE_COPY',flush=True)
    while time.monotonic()<deadline:
        try:
            manifest=json.loads((root/'verified_manifest.json').read_text())
            if manifest['repo']!=REPO or manifest['revision']!=REVISION or not manifest['weights_verified']:
                raise ValueError('Wrong cache revision')
            expected=json.loads((Path(__file__).resolve().parents[1]/'docs/apertus-v15-model-metadata.json').read_text())['weights']
            entries={entry['name']:entry for entry in manifest['files']}
            if any(entries.get(entry['name'],{}).get('sha256')!=entry['sha256'] for entry in expected):
                raise ValueError('Pinned original weight shards missing from cache manifest')
            if all((root/entry['name']).is_file() and (root/entry['name']).stat().st_size==entry['bytes'] for entry in manifest['files']):
                for entry in manifest['files']:
                    if digest(root/entry['name'])!=entry['sha256']:raise ValueError('Transferred model checksum mismatch')
                print('V15_RECOVERED_MODEL_CACHE_VERIFIED '+REVISION,flush=True)
                return
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(2)
    raise TimeoutError('No verified complete model cache received before transfer deadline')


if __name__=='__main__':main()
