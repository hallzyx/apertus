"""Download the pinned public Apertus model and verify official LFS hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from huggingface_hub import HfApi, snapshot_download
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    repo = 'swiss-ai/Apertus-8B-Instruct-2509'
    revision = 'b946d40447b2b597999b9c86d44bee0b452c919f'
    print('Checking pinned official Apertus model metadata', flush=True)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True, timeout=30)
    if info.sha != revision:
        raise SystemExit('Official model revision mismatch')
    print('Downloading original model shards and official tokenizer/template', flush=True)
    root = Path(snapshot_download(repo, revision=revision, local_dir=args.model_dir,
                                 allow_patterns=['*.json', '*.safetensors', '*.txt',
                                                 '*.model', 'chat_template.jinja']))
    files = []
    for item in info.siblings:
        if not item.rfilename.endswith('.safetensors'):
            continue
        if item.lfs is None:
            raise SystemExit('Missing official model shard checksum')
        print('Verifying model shard:', item.rfilename, flush=True)
        path = root / item.rfilename
        digest = hashlib.sha256()
        with path.open('rb') as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        if path.stat().st_size != item.lfs.size or digest.hexdigest() != item.lfs.sha256:
            raise SystemExit('Model shard integrity check failed')
        files.append({'name': item.rfilename, 'size': item.lfs.size,
                      'sha256': item.lfs.sha256})
    if len(files) != 4 or not (root / 'chat_template.jinja').is_file():
        raise SystemExit('Incomplete pinned model or missing official chat template')
    manifest = {'repo': repo, 'revision': revision, 'files': files,
                'weights_verified': True}
    (root / 'verified_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Pinned Apertus model and all four official weight checksums verified')


if __name__ == '__main__':
    main()
