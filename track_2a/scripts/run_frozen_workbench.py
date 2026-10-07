"""Run original Apertus and the web app inside one reproducible CPU container."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',default=os.environ.get('APERTUS_MODEL_DIR','/models/apertus-8b'))
    p.add_argument('--head-dir',default='/app/deployment')
    p.add_argument('--predict',nargs=2,metavar=('BOOKLET','CLAIM'))
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        raise SystemExit(subprocess.call([sys.executable,'-m','ost_nli','self-test']))
    if not (Path(args.model_dir)/'verified_manifest.json').exists():
        subprocess.run([sys.executable,'scripts/download_cpu_model.py','--model-dir',args.model_dir],check=True)
    if args.predict:
        subprocess.run([sys.executable,'-m','ost_nli','predict-frozen',*args.predict,
                        '--model-dir',args.model_dir,'--head-dir',args.head_dir],check=True)
        return
    child=subprocess.Popen([sys.executable,'scripts/serve_frozen_apertus.py','--model-dir',args.model_dir,
                            '--head-dir',args.head_dir,'--host','127.0.0.1','--port','8001'])
    def stop(signum=None,frame=None):
        child.terminate()
        if signum is not None:raise SystemExit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for attempt in range(360):
            if child.poll() is not None:raise RuntimeError('Real model backend failed during startup')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8001/health',timeout=3) as response:
                    if json.load(response)['status']=='ready':break
            except OSError:time.sleep(1)
        else:raise RuntimeError('Real model backend startup timed out')
        os.environ['FROZEN_BASE_URL']='http://127.0.0.1:8001'
        from ost_nli.web import serve
        serve('0.0.0.0',8000)
    finally:
        stop();child.wait(timeout=30)


if __name__=='__main__':main()
