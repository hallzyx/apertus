"""Retrieve only a specified authorized Vast instance's logs; never print keys."""
import argparse
import base64
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    key = os.environ.get('VAST_API_KEY')
    if not key:
        raise SystemExit('Secure VAST_API_KEY binding missing')
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}
    request = urllib.request.Request(
        f'https://console.vast.ai/api/v0/instances/request_logs/{args.instance}/',
        headers=headers, data=b'{"tail":"20000"}', method='PUT')
    with urllib.request.urlopen(request, timeout=30) as response:
        task = json.load(response)
    url = task.get('result_url')
    if not url or not url.startswith('https://s3.amazonaws.com/'):
        raise SystemExit('Unexpected or unavailable log destination; inspect host only')
    text = None
    for attempt in range(10):
        try:
            # Intentionally no Authorization header on the S3 request.
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(100_000_000)
            text = raw.decode('utf-8', errors='replace')
            break
        except urllib.error.HTTPError as error:
            if error.code not in (403, 404):
                raise
            time.sleep(1)
    if text is None:
        raise SystemExit('Log result object not available after bounded upload wait')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'latest.log').write_text(text)
    digest = re.findall(r'APERTUS_RESULTS_SHA256=([0-9a-f]{64})', text)
    payloads = re.findall(r'APERTUS_RESULTS_B64=([A-Za-z0-9+/=]+)', text)
    if digest and payloads:
        data = base64.b64decode(payloads[-1], validate=True)
        if hashlib.sha256(data).hexdigest() != digest[-1]:
            raise SystemExit('Remote artifact checksum mismatch')
        (root / 'results.tar.gz').write_bytes(data)
        (root / 'artifact_integrity.json').write_text(json.dumps(
            {'instance_id': args.instance, 'sha256': digest[-1], 'bytes': len(data),
             'verified': True}, indent=2) + '\n')
        print('Verified artifact archive recovered:', len(data), 'bytes')
    lines = [line for line in text.splitlines() if 'APERTUS_RESULTS_B64=' not in line]
    print('\n'.join(lines[-12:]))


if __name__ == '__main__':
    main()
