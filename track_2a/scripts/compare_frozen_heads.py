"""Recompute validation comparisons and portable-head parity without test access."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import numpy as np
    from ost_nli.data import fingerprint, load_dataset
    from ost_nli.frozen import head_probabilities
    from ost_nli.metrics import evaluate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--experiments', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    rows = load_dataset(args.validation)
    experiments, cache = Path(args.experiments), Path(args.cache)
    source = json.loads((cache / 'experiment.json').read_text())
    if source['status'] != 'completed' or source['validation_sha256'] != fingerprint(rows):
        raise ValueError('Completed fingerprint-matched cache required')
    for entry in source['files']:
        if hashlib.sha256((cache / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Cache checksum mismatch')
    unique, seen = [], set()
    for row in rows:
        key = (row['claim'], '\n\n'.join(p['text'] for p in row['passages']),
               row['claim_language'], row['document_language'], row['label'])
        if key not in seen:
            unique.append(row)
            seen.add(key)
    results = []
    for cap in [1024, 4096]:
        names = [f'apertus-option-head-{cap}-v1'] + [
            f'apertus-frozen-{kind}-{cap}-v1' for kind in ['option', 'hidden']]
        for name in names:
            directory = experiments / name
            record = json.loads((directory / 'experiment.json').read_text())
            bundle = json.loads((directory / 'predictions.json').read_text())
            if record['status'] != 'completed' or bundle['dataset_sha256'] != fingerprint(rows):
                raise ValueError('Complete matching predictions required: ' + name)
            predictions = bundle['predictions']
            metrics = evaluate(rows, predictions)
            if abs(metrics['macro_f1'] - record['macro_f1']) > 1e-12:
                raise ValueError('Recomputed score mismatch: ' + name)
            unique_ids = {row['id'] for row in unique}
            dedup = evaluate(unique, [r for r in predictions if r['id'] in unique_ids])
            parity = None
            if name.startswith('apertus-frozen-'):
                parameters = json.loads((directory / 'head.json').read_text())
                with np.load(cache / str(cap) / 'validation.npz', allow_pickle=False) as values:
                    if values['ids'].tolist() != [row['id'] for row in rows]:
                        raise ValueError('Cache row alignment mismatch')
                    computed = [head_probabilities(vector, parameters) for vector in values[parameters['feature_kind']]]
                maximum = float(np.max(np.abs(np.asarray(computed) - np.asarray([r['probabilities'] for r in predictions]))))
                if maximum > 1e-10:
                    raise ValueError('Portable deployment probability mismatch: ' + name)
                parity = {'n': len(rows), 'max_absolute_probability_error': maximum, 'passed': True}
            results.append({'experiment_id': name, 'reference_token_cap': cap,
                            'training_n': record['training_configuration']['n'],
                            'metrics': metrics, 'deduplicated_metrics': dedup,
                            'portable_probability_parity': parity})
    selected = max(results, key=lambda r: (r['metrics']['macro_f1'], -r['reference_token_cap']))
    output = {'validation_sha256': fingerprint(rows), 'n': len(rows),
              'n_exact_deduplicated': len(unique), 'official': False,
              'test_evaluated': False, 'selection': 'Maximum internal validation Macro-F1; shorter cap breaks ties',
              'selected_experiment_id': selected['experiment_id'], 'results': results,
              'limitations': 'Two independent validation components; two training components. Supplied references, unresolved official premise scope/class ontology. No final-test claims.'}
    Path(args.output).write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    for r in results:
        print(r['experiment_id'], 'Macro-F1', r['metrics']['macro_f1'],
              'deduplicated', r['deduplicated_metrics']['macro_f1'], flush=True)
    print('Selected:', selected['experiment_id'], flush=True)


if __name__ == '__main__':
    main()
