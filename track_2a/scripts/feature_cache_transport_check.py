"""Offline integrity/fault check for the cache transport, without Vast calls."""
import base64
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

exporter = Path(__file__).with_name('export_vast_cache.py').resolve()
# Exporter uses absolute /workspace files; exercise its format without mutating those paths.
source = exporter.read_text().replace("Path('/workspace/feature-cache.tar.gz')", "Path('feature-cache.tar.gz')").replace("Path('/workspace/cache-export')", "Path('cache-export')")
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    cache = root / 'track_2a/experiments/apertus-frozen-cache-v1'
    cache.mkdir(parents=True)
    content = os.urandom(600_000)
    (cache / 'example.bin').write_bytes(content)
    (cache / 'experiment.json').write_text('{"status":"completed"}')
    script = root / 'export.py'; script.write_text(source)
    subprocess.run(['python', str(script)], cwd=root, check=True, capture_output=True)
    manifest = json.loads((root / 'cache-export/manifest.json').read_text())
    parts = []
    for entry in manifest['parts']:
        data = (root / 'cache-export' / entry['name']).read_bytes()
        assert len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert max(map(len, data.splitlines())) <= 400
        assert hashlib.sha256(data[:-1]).hexdigest() != entry['sha256'], 'Truncation must fail'
        parts.append(b''.join(data.split()))
    archive = base64.b64decode(b''.join(parts), validate=True)
    assert hashlib.sha256(archive).hexdigest() == manifest['archive_sha256']
    assert len(archive) == manifest['archive_bytes']
print('Cache transport source SHA-256 round trip and truncation detection passed')
