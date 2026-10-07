"""Restore the exact predeclared training IDs after a source/cache transfer."""
import argparse
import json
from pathlib import Path


def main():
    from ost_nli.data import fingerprint, load_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-dir', required=True)
    parser.add_argument('--inputs', required=True)
    args = parser.parse_args()
    full = Path(args.full_dir)
    manifest = json.loads((Path(args.inputs) / 'manifest.json').read_text())
    for split in ['validation', 'test']:
        if fingerprint(load_dataset(full / f'{split}.jsonl')) != manifest['splits'][split]['full_sha256']:
            raise ValueError('Complete frozen validation/test fingerprint mismatch')
    train_path = full / 'train.jsonl'
    train = load_dataset(train_path)
    expected = manifest['splits']['train']['full_sha256']
    if fingerprint(train) != expected:
        policy = manifest.get('training_source_policy')
        if not policy:
            raise ValueError('No declared training subset policy')
        by_id = {row['id']: row for row in train}
        selected = [by_id[row_id] for row_id in policy['selected_ids']]
        if fingerprint(selected) != expected:
            raise ValueError('Transferred training source changed frozen rows')
        backup = full / 'train-source-observed-after-repair.jsonl'
        if backup.exists():
            raise ValueError('Existing source-observation backup cannot be overwritten')
        train_path.rename(backup)
        train_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
        (full / 'frozen-training-recovery.json').write_text(json.dumps({
            'observed_n': len(train), 'frozen_n': len(selected),
            'frozen_sha256': expected, 'selection': 'Exact IDs fixed before inference; later source availability cannot change training selection',
            'validation_test_unchanged': True}, indent=2) + '\n')
    print('V15_ALL_FROZEN_SOURCE_ROWS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
