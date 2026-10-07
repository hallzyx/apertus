"""Wait for a task-started CPU run, preserve its journal, and push its artifacts.

Does not restart inference, rent hardware, rewrite history, or stage other work.
Snapshot restoration stops live processes; this helper must not be assumed alive.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--legacy-registry')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    run = Path(args.run_dir).resolve()
    if run.parent != repo / 'track_2a' / 'experiments':
        raise SystemExit('Run must be inside the task experiment directory')
    while True:
        record = json.loads((run / 'experiment.json').read_text())
        if record['status'] in ('completed', 'failed', 'interrupted'):
            break
        try:
            os.kill(args.pid, 0)
            command = Path(f'/proc/{args.pid}/cmdline').read_bytes()
            if b'cpu_ost_experiment.py' not in command:
                raise ProcessLookupError('Task process no longer matches')
        except (ProcessLookupError, FileNotFoundError):
            # Give the worker a chance to finish its atomic-sized final writes.
            time.sleep(1)
            record = json.loads((run / 'experiment.json').read_text())
            if record['status'] == 'started':
                record = {**record, 'status': 'interrupted',
                          'notes': record['notes'] + '; worker exited before final record; partial scores only'}
                (run / 'experiment.json').write_text(json.dumps(record, indent=2) + '\n')
            break
        time.sleep(10)
    registry = repo / 'track_2a' / 'experiments' / 'registry.jsonl'
    events = [json.loads(line) for line in registry.read_text().splitlines()]
    extras = [record]
    if args.legacy_registry:
        old = Path(args.legacy_registry).resolve()
        if old != repo / 'experiments' / 'registry.jsonl':
            raise SystemExit('Unexpected legacy registry path')
        extras = [json.loads(line) for line in old.read_text().splitlines()] + extras
    with registry.open('a') as output:
        for event in extras:
            if event not in events:
                output.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
                events.append(event)
    if record['status'] == 'completed':
        from ost_nli.data import fingerprint, load_dataset
        from ost_nli.metrics import evaluate
        rows = load_dataset(repo / 'track_2a/data/private/splits-strict/validation.jsonl')
        bundle = json.loads((run / 'predictions.json').read_text())
        if fingerprint(rows) != bundle['dataset_sha256']:
            raise SystemExit('Validation fingerprint changed; refusing to publish')
        metrics = evaluate(rows, bundle['predictions'])
        if metrics['n'] != 276 or abs(metrics['macro_f1'] - record['macro_f1']) > 1e-12:
            raise SystemExit('Completed metric verification failed')
    paths = [str(run.relative_to(repo)), str(registry.relative_to(repo))]
    subprocess.run(['git', 'add', '--', *paths], cwd=repo, check=True)
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', *paths], cwd=repo)
    if changed.returncode == 1:
        subprocess.run(['git', '-c', 'user.name=Codex', '-c',
                        'user.email=codex@users.noreply.github.com', 'commit', '--only',
                        '-m', 'Persist real Apertus CPU run outcome', '--', *paths],
                       cwd=repo, check=True)
    elif changed.returncode != 0:
        raise SystemExit('Git diff failed')
    subprocess.run(['git', 'push', 'origin', 'HEAD:main'], cwd=repo, check=True)
    print('CPU run outcome durably pushed:', record['status'], flush=True)


if __name__ == '__main__':
    main()
