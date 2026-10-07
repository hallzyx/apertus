"""Preserve completed native validation/train features before head dependencies."""
import hashlib,json,shutil,subprocess
from pathlib import Path

work=Path('/workspace');repo=Path(__file__).resolve().parents[2]
root=repo/'track_2a/experiments/apertus-v15-features-ready-v1'
if not root.exists():
    root.mkdir()
    for name in ('validation','train'):
        source=work/f'v15-{name}'
        assert json.loads((source/'experiment.json').read_text())['status']=='completed'
        shutil.copytree(source,root/name)
    for name,source in [('model_manifest.json',work/'model/verified_manifest.json'),('inputs_manifest.json',work/'v15-inputs/manifest.json')]:
        shutil.copyfile(source,root/name)
    files=[{'path':str(f.relative_to(root)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in sorted(root.rglob('*')) if f.is_file()]
    (root/'experiment.json').write_text(json.dumps({'status':'completed','stage':'validation and training features; no fitted heads or test inference',
        'model':'swiss-ai/Apertus-v1.5-8B','model_revision':'a411d838600baf0e3635a3daf66fb7c55fc97bb6','files':files},indent=2)+'\n')
subprocess.run(['python',str(repo/'track_2a/scripts/export_vast_cache.py'),'--relative-root',str(root.relative_to(repo))],cwd=repo,check=True)
print('V15_FEATURE_CHECKPOINT_READY',flush=True)
