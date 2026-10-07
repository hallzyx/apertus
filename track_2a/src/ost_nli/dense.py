"""Pinned multilingual E5 retrieval; supporting model, never NLI classifier."""
import hashlib
import json
from pathlib import Path

REPO = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'


def download(directory):
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION: raise ValueError('Embedding revision mismatch')
    root = Path(snapshot_download(REPO, revision=REVISION, token=False, local_dir=directory,
                                 allow_patterns=['*.json', 'model.safetensors', '*.model']))
    files = []
    for item in info.siblings:
        if item.lfs and (root / item.rfilename).is_file():
            path = root / item.rfilename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item.lfs.sha256 or path.stat().st_size != item.lfs.size:
                raise ValueError('Embedding source checksum mismatch')
            files.append({'name': item.rfilename, 'sha256': digest})
    if not any(r['name'] == 'model.safetensors' for r in files):
        raise ValueError('Embedding weights missing')
    (root / 'verified_manifest.json').write_text(json.dumps({'repo': REPO, 'revision': REVISION, 'files': files}, indent=2) + '\n')
    return root


class MultilingualRetriever:
    def __init__(self, directory, cache_dir=None):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch; torch.set_num_threads(4)
        root = Path(directory); manifest = json.loads((root / 'verified_manifest.json').read_text())
        if manifest['repo'] != REPO or manifest['revision'] != REVISION:
            raise ValueError('Pinned embedding manifest required')
        for entry in manifest['files']:
            if hashlib.sha256((root / entry['name']).read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Embedding checksum mismatch')
        self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(root, local_files_only=True, trust_remote_code=False).eval()
        self.documents = {}; self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir: self.cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(self, texts, prefix):
        import numpy as np
        result = []
        for start in range(0, len(texts), 16):
            inputs = self.tokenizer([prefix + t for t in texts[start:start+16]], padding=True,
                                    truncation=True, max_length=512, return_tensors='pt')
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1)
                result.append(self.torch.nn.functional.normalize(pooled, p=2, dim=1).numpy())
        return np.concatenate(result)

    def retrieve(self, passages, claim, k=5, mode='hybrid'):
        import numpy as np
        from .retrieval import retrieve
        if k < 1 or mode not in ('dense', 'hybrid'): raise ValueError('Invalid retrieval settings')
        key = hashlib.sha256(json.dumps([(p['id'], p['text']) for p in passages], ensure_ascii=False).encode()).hexdigest()
        if key not in self.documents:
            path = self.cache_dir / (key + '.npz') if self.cache_dir else None
            if path and path.exists():
                with np.load(path, allow_pickle=False) as values:
                    if values['revision'].item() != REVISION: raise ValueError('Stale embedding cache')
                    vectors = values['vectors']
            else:
                vectors = self.encode([p['text'] for p in passages], 'passage: ')
                if path: np.savez_compressed(path, vectors=vectors, revision=np.asarray(REVISION))
            if vectors.shape != (len(passages), 384) or not np.isfinite(vectors).all():
                raise ValueError('Invalid document embeddings')
            self.documents[key] = vectors
        scores = self.documents[key] @ self.encode([claim], 'query: ')[0]
        dense_order = sorted(range(len(passages)), key=lambda i: (-float(scores[i]), i))
        fused = {i: 1 / (60 + rank) for rank, i in enumerate(dense_order, 1)}
        if mode == 'hybrid':
            index = {p['id']: i for i, p in enumerate(passages)}
            for rank, p in enumerate(retrieve(passages, claim, k=len(passages)), 1):
                if p['retrieval_score'] > 0:
                    fused[index[p['id']]] += 1 / (60 + rank)
        order = dense_order if mode == 'dense' else sorted(fused, key=lambda i: (-fused[i], i))
        return [{**passages[i], 'retrieval_score': float(scores[i]), 'retrieval_method': mode,
                 'fusion_score': fused[i], 'embedding_model': REPO, 'embedding_revision': REVISION} for i in order[:k]]
