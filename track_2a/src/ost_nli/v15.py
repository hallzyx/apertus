"""Official pinned Apertus v1.5 engine; requires the native Swiss AI fork."""
import hashlib
import json
import os
import time
from pathlib import Path

REPO = 'swiss-ai/Apertus-v1.5-8B'
REVISION = 'a411d838600baf0e3635a3daf66fb7c55fc97bb6'
TRANSFORMERS_REVISION = '3797303dda74844e3d1f8977ff5518bb91f818b4'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


def download(directory, token, external_shard_dir=None):
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    if not token: raise ValueError('Authorized read token required for first download')
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, token=token)
    if info.sha != REVISION: raise ValueError('Model revision mismatch')
    ignore = []
    if external_shard_dir:
        # A split cache permits small cloud disks without copying a large shard.
        name='model-apertus-model-00002-of-00004.safetensors'
        external=Path(hf_hub_download(REPO,name,revision=REVISION,token=token,local_dir=external_shard_dir))
        Path(directory).mkdir(parents=True,exist_ok=True)
        link=Path(directory)/name
        if not link.exists():link.symlink_to(external)
        if link.resolve()!=external.resolve():raise ValueError('Unexpected external shard target')
        ignore=[name]
    root = Path(snapshot_download(REPO, revision=REVISION, token=token, local_dir=directory,
        allow_patterns=['*.json','*.safetensors','*.jinja','*.model','*.txt','README.md'],
        ignore_patterns=ignore,max_workers=4))
    expected = json.loads((Path(__file__).resolve().parents[2]/'docs/apertus-v15-model-metadata.json').read_text())
    files = []
    for item in info.siblings:
        path = root/item.rfilename
        if not path.is_file() or '.cache/' in item.rfilename: continue
        sha = digest(path)
        if item.lfs and (sha != item.lfs.sha256 or path.stat().st_size != item.lfs.size):
            raise ValueError('Model source checksum mismatch')
        files.append({'name':item.rfilename,'sha256':sha,'bytes':path.stat().st_size})
    actual = {f['name']:f for f in files}
    for entry in expected['weights']:
        if actual.get(entry['name'],{}).get('sha256') != entry['sha256']:
            raise ValueError('Pinned weight shard missing or mismatched')
    manifest = {'repo':REPO,'revision':REVISION,'weights_verified':True,'files':files,
        'transformers_revision':TRANSFORMERS_REVISION,'token_persisted':False}
    (root/'verified_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class Engine:
    def __init__(self, directory, device='cuda'):
        import torch
        import transformers
        from transformers import AutoTokenizer, Apertus1p5ForConditionalGeneration
        root = Path(directory)
        manifest = json.loads((root/'verified_manifest.json').read_text())
        if manifest['repo'] != REPO or manifest['revision'] != REVISION or not manifest['weights_verified']:
            raise ValueError('Pinned verified v1.5 weights required')
        for entry in manifest['files']:
            if digest(root/entry['name']) != entry['sha256']: raise ValueError('Model checksum failure')
        if device not in ('cpu','cuda'):raise ValueError('Unsupported inference device')
        if device=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise ValueError('This runtime requires BF16 CUDA hardware')
        torch.set_num_threads(4); torch.manual_seed(42)
        self.torch = torch; self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.model, loading = Apertus1p5ForConditionalGeneration.from_pretrained(root,
            local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,
            attn_implementation={'':'eager','text_config':'sdpa',
                'vision_tokenizer_config':'eager','audio_tokenizer_config':'eager'},output_loading_info=True)
        if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise ValueError('Incomplete or mismatched native model loading')
        self.model = self.model.to(device).eval()
        self.metadata = {'repo':REPO,'revision':REVISION,'transformers_revision':TRANSFORMERS_REVISION,
            'torch':torch.__version__,'transformers':transformers.__version__,
            'gpu':torch.cuda.get_device_name(0) if device=='cuda' else 'CPU',
            'gpu_bytes':torch.cuda.get_device_properties(0).total_memory if device=='cuda' else None,
            'precision':'BF16 text; official tokenizer modules remain FP32','quantization':None}
        self.numeric = [self.tokenizer.encode(str(i),add_special_tokens=False) for i in range(3)]
        if any(len(x)!=1 for x in self.numeric): raise ValueError('Numeric labels require single-token options')

    def infer(self, messages, method='score', head=None):
        if method == 'head' and (not head or head.get('feature_kind') not in ('hidden','option_logits')):
            raise ValueError('Single-forward Engine requires a hidden or option_logits head; paired research heads need an explicit paired runtime')
        import numpy as np
        torch = self.torch
        began = time.perf_counter()
        text = self.tokenizer.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        if method in ['score','head']: text += '{"label":'
        inputs = self.tokenizer(text,return_tensors='pt',add_special_tokens=False).to(self.device)
        tokens = int(inputs['input_ids'].shape[1])
        if tokens > 32768: raise ValueError('Prompt exceeds documented experiment/runtime token limit')
        hidden = None; scores = None; probabilities = None; invalid = False
        if self.device=='cuda':torch.cuda.synchronize()
        if method == 'prompt':
            with torch.inference_mode():
                output = self.model.generate(**inputs,max_new_tokens=32,do_sample=False)
            content = self.tokenizer.decode(output[0,tokens:],skip_special_tokens=True)
            try:
                result = json.loads(content)
                if set(result) != {'label'} or type(result['label']) is not int or result['label'] not in (0,1,2): raise ValueError()
                label = result['label']
            except (ValueError,TypeError): label = None; invalid = True
            completion_tokens = int(output.shape[1]-tokens)
        else:
            captured = []
            hook = self.model.model.language_model.norm.register_forward_hook(
                lambda module,args,result:captured.append(result[0,-1].detach().float().cpu().numpy().copy()))
            try:
                with torch.inference_mode():
                    output = self.model(**inputs,use_cache=False,logits_to_keep=1)
                hidden = captured[0]
                scores = output.logits[0,-1,[x[0] for x in self.numeric]].float().cpu().numpy().copy()
                if not np.isfinite(hidden).all() or not np.isfinite(scores).all(): raise ValueError('Nonfinite Apertus features')
                if method == 'head':
                    x = scores-scores.mean() if head['feature_kind']=='option_logits' else hidden
                    scaled = (x-np.asarray(head['feature_mean']))/np.asarray(head['feature_scale'])
                    scores_for_prob = (np.asarray(head['coefficients'])@scaled+np.asarray(head['intercept']))/head['temperature']
                else: scores_for_prob = scores
                exp = np.exp(scores_for_prob-scores_for_prob.max()); probabilities = (exp/exp.sum()).tolist()
                label = int(np.argmax(probabilities)); content = json.dumps({'label':label}); completion_tokens = 1
            finally: hook.remove()
        if self.device=='cuda':torch.cuda.synchronize()
        return {'label':label,'content':content,'invalid_output':invalid,'probabilities':probabilities,
            'probability_note':'Restricted next-token class probabilities; uncalibrated' if method=='score' else
                ('Training-group OOF calibrated head posterior' if method=='head' else 'Unavailable'),
            'context_tokens':tokens,'completion_tokens':completion_tokens,
            'latency_seconds':time.perf_counter()-began,'hidden':hidden,'option_logits':scores}
