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
    chunks = re.findall(r'APERTUS_RESULTS_CHUNK=(\d+):([A-Za-z0-9+/=]+)',
                        text.rsplit('APERTUS_RESULTS_SHA256=', 1)[-1])
    if digest and chunks:
        parts = {int(index): value for index, value in chunks}
        if len(parts) != len(chunks) or sorted(parts) != list(range(len(parts))):
            raise SystemExit('Missing, duplicated or truncated archive chunks')
        data = base64.b64decode(''.join(parts[index] for index in range(len(parts))), validate=True)
        if hashlib.sha256(data).hexdigest() != digest[-1]:
            raise SystemExit('Remote chunked artifact checksum mismatch')
    elif digest and payloads:
        # Docker logging can split a long stdout line into multiple records.
        suffix = text.rsplit('APERTUS_RESULTS_B64=', 1)[1]
        joined = ''
        data = None
        for line in suffix.splitlines():
            if not re.fullmatch(r'[A-Za-z0-9+/=]+', line):
                break
            joined += line
            if len(joined) % 4:
                continue
            candidate = base64.b64decode(joined, validate=True)
            if hashlib.sha256(candidate).hexdigest() == digest[-1]:
                data = candidate
                break
        if data is None:
            raise SystemExit('Remote artifact checksum mismatch')
    else:
        data = None
    if data is not None:
        (root / 'results.tar.gz').write_bytes(data)
        (root / 'artifact_integrity.json').write_text(json.dumps(
            {'instance_id': args.instance, 'sha256': digest[-1], 'bytes': len(data),
             'verified': True}, indent=2) + '\n')
        print('Verified artifact archive recovered:', len(data), 'bytes')
    lines = [line for line in text.splitlines() if 'APERTUS_RESULTS_B64=' not in line
             and 'APERTUS_RESULTS_CHUNK=' not in line
             and not (len(line) > 150 and re.fullmatch(r'[A-Za-z0-9+/=]+', line))]
    print('\n'.join(lines[-12:]))


if __name__ == '__main__':
    main()
