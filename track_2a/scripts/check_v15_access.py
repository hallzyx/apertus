"""Read-only authenticated preflight; never download weights or print credentials."""
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = 'swiss-ai/Apertus-v1.5-8B'
REVISION = 'a411d838600baf0e3635a3daf66fb7c55fc97bb6'


class ScopedRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('Model download redirects must remain HTTPS')
        result = super().redirect_request(request, response, code, message, headers, newurl)
        if result and urllib.parse.urlsplit(newurl).hostname != 'huggingface.co':
            result.remove_header('Authorization')
        return result


def main():
    token = os.environ.get('HF_TOKEN')
    if not token:
        print(json.dumps({'repo':REPO,'revision':REVISION,'auth_configured':False,
                          'status':'missing_secure_HF_TOKEN','weights_downloaded':False}))
        return 2
    opener = urllib.request.build_opener(ScopedRedirect())
    url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/config.json'
    try:
        request = urllib.request.Request(url, headers={'Authorization':'Bearer '+token})
        with opener.open(request,timeout=30) as response:
            content = response.read(1_000_001)
        if len(content)>1_000_000: raise ValueError('Unexpected oversized config')
        config = json.loads(content)
        if config.get('model_type')!='apertus1p5': raise ValueError('Wrong model generation')
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        supports_native = 'apertus1p5' in CONFIG_MAPPING
        output = Path('/workspace/.cache/apertus-v15-authorized')
        output.mkdir(exist_ok=True)
        (output/'config.json').write_bytes(content)
        readme = f'https://huggingface.co/{REPO}/resolve/{REVISION}/README.md'
        with opener.open(urllib.request.Request(readme,headers={'Authorization':'Bearer '+token}),timeout=30) as response:
            (output/'README.md').write_bytes(response.read(1_000_000))
        print(json.dumps({'repo':REPO,'revision':REVISION,'auth_configured':True,
                          'config_access_verified':True,'config_sha256':hashlib.sha256(content).hexdigest(),
                          'installed_transformers_native_support':supports_native,
                          'weights_downloaded':False,'gpu_rented':False}))
        return 0
    except urllib.error.HTTPError as error:
        print(json.dumps({'repo':REPO,'revision':REVISION,'auth_configured':True,
                          'config_access_verified':False,'http_status':error.code,
                          'weights_downloaded':False}))
        return 2


if __name__ == '__main__': raise SystemExit(main())
