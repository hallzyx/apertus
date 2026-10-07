"""Package task feature data into checksummed UTF-8 parts for Vast cat reads."""
import base64
import hashlib
import json
import tarfile
from pathlib import Path


root = Path('track_2a/experiments/apertus-frozen-cache-v1')
if not root.exists():
    print('APERTUS_CACHE_MISSING=1', flush=True)
else:
    archive = Path('/workspace/feature-cache.tar.gz')
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.add(root, arcname=str(root))
    data = archive.read_bytes()
    encoded = base64.b64encode(data)
    output = Path('/workspace/cache-export')
    output.mkdir(exist_ok=True)
    parts = []
    for index, offset in enumerate(range(0, len(encoded), 400_000)):
        part = encoded[offset:offset+400_000]
        name = f'part-{index:04d}.b64'
        content = b'\n'.join(part[i:i+400] for i in range(0, len(part), 400)) + b'\n'
        (output / name).write_bytes(content)
        parts.append({'name': name, 'bytes': len(content),
                      'sha256': hashlib.sha256(content).hexdigest()})
    manifest = {'archive_sha256': hashlib.sha256(data).hexdigest(),
                'archive_bytes': len(data), 'parts': parts}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('APERTUS_CACHE_READY=' + json.dumps({'archive_bytes': len(data),
          'parts': len(parts), 'sha256': manifest['archive_sha256']}), flush=True)
