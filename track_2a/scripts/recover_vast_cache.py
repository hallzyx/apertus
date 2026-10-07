"""Recover a task's stopped-instance feature cache with source SHA-256 checks."""
import argparse
import base64
import hashlib
import io
import json
import os
import re
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance', required=True, type=int)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--checkout', required=True)
    args = p.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    headers = {'Authorization': 'Bearer ' + os.environ['VAST_API_KEY'], 'Content-Type': 'application/json'}
    def read_remote(path):
        req = urllib.request.Request(f'https://console.vast.ai/api/v0/instances/command/{args.instance}/',
              headers=headers, data=json.dumps({'command': 'cat ' + path}).encode(), method='PUT')
        with urllib.request.urlopen(req, timeout=40) as response:
            result = json.load(response)
        url = result.get('result_url')
        if not url or not url.startswith('https://s3.amazonaws.com/'):
            raise ValueError('Unsupported result destination')
        for attempt in range(20):
            try:
                with urllib.request.urlopen(url, timeout=40) as response:
                    return response.read(1_000_000)
            except urllib.error.HTTPError as error:
                if error.code not in (403, 404):
                    raise
                time.sleep(1)
        raise RuntimeError('Result upload unavailable after bounded wait')
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists():
        manifest_path.write_bytes(read_remote('/workspace/cache-export/manifest.json'))
    manifest = json.loads(manifest_path.read_text())
    encoded = []
    for entry in manifest['parts']:
        name = entry['name']
        if not re.fullmatch(r'part-\d{4}\.b64', name):
            raise ValueError('Invalid part name')
        local = root / name
        if not local.exists():
            local.write_bytes(read_remote('/workspace/cache-export/' + name))
        content = local.read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('Remote part checksum mismatch: ' + name)
        encoded.append(b''.join(content.split()))
        print('Verified cache part', name, len(content), 'bytes', flush=True)
    data = base64.b64decode(b''.join(encoded), validate=True)
    if len(data) != manifest['archive_bytes'] or hashlib.sha256(data).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Original remote archive checksum mismatch')
    checkout = Path(args.checkout).resolve()
    prefix = 'track_2a/experiments/apertus-frozen-cache-v1'
    target = checkout / prefix
    if target.exists() and any(target.iterdir()):
        raise ValueError('Existing feature cache cannot be overwritten')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as bundle:
        for member in bundle.getmembers():
            resolved = (checkout / member.name).resolve()
            if member.name != prefix and not member.name.startswith(prefix + '/'):
                raise ValueError('Unexpected archive member')
            if not resolved.is_relative_to(target) or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive member')
        bundle.extractall(checkout, filter='data')
    source = json.loads((target / 'experiment.json').read_text())
    for entry in source.get('files', []):
        path = (target / entry['path']).resolve()
        if not path.is_relative_to(target) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Feature file checksum mismatch')
    proof = {'instance_id': args.instance, 'archive_sha256': manifest['archive_sha256'],
             'archive_bytes': len(data), 'part_count': len(manifest['parts']),
             'original_remote_archive_verified': True, 'cache_status': source['status']}
    (root / 'artifact_integrity.json').write_text(json.dumps(proof, indent=2) + '\n')
    print('Source checksums verified; cache status:', source['status'], flush=True)


if __name__ == '__main__':
    main()
