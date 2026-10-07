"""Verify paired Apertus validation artifacts and compare reference token caps."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    from ost_nli.data import fingerprint, load_dataset, words
    from ost_nli.metrics import classification, evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--run-a', required=True)
    parser.add_argument('--run-b', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = load_dataset(args.validation)
    runs = []
    for directory in [args.run_a, args.run_b]:
        root = Path(directory)
        record = json.loads((root / 'experiment.json').read_text())
        bundle = json.loads((root / 'predictions.json').read_text())
        if record['status'] != 'completed' or bundle['dataset_sha256'] != fingerprint(rows):
            raise SystemExit('Completed fingerprint-aligned runs required')
        metrics = evaluate(rows, bundle['predictions'])
        if abs(metrics['macro_f1'] - record['macro_f1']) > 1e-12:
            raise SystemExit('Recorded score differs from recomputed predictions')
        runs.append((record, metrics, {p['id']: p for p in bundle['predictions']}))
    a, b = runs
    for field in ['model', 'model_revision', 'precision', 'training_mapping_ids']:
        if a[0][field] != b[0][field]:
            raise SystemExit('Unmatched experiment condition: ' + field)
    # Cluster dates connected by the same normalized claim, preserving dependence.
    parent = {row['booklet_id']: row['booklet_id'] for row in rows}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    seen = {}
    for row in rows:
        claim = ' '.join(words(row['claim']))
        if claim in seen:
            parent[find(row['booklet_id'])] = find(seen[claim])
        else:
            seen[claim] = row['booklet_id']
    groups = defaultdict(list)
    for row in rows:
        groups[find(row['booklet_id'])].append(row)
    clusters = list(groups.values())
    interval = None
    if len(clusters) > 1:
        rng = random.Random(42)
        deltas = []
        for _ in range(5000):
            sample = [row for _ in clusters for row in rng.choice(clusters)]
            truth = [row['label'] for row in sample]
            scores = [classification(truth, [run[2][row['id']]['label'] for row in sample])['macro_f1']
                      for run in runs]
            deltas.append(scores[1] - scores[0])
        deltas.sort()
        interval = [deltas[124], deltas[4874]]
    summaries = []
    for record, metrics, predictions in runs:
        summaries.append({'experiment_id': record['experiment_id'], 'macro_f1': metrics['macro_f1'],
                          'accuracy': metrics['accuracy'], 'per_class': metrics['per_class'],
                          'by_language': metrics['by_language'], 'cross_lingual': metrics['cross_lingual'],
                          'average_context_tokens': metrics['average_context_tokens'],
                          'average_latency_seconds': metrics['average_latency_seconds'],
                          'truncated_examples': record['n_truncated'],
                          'numeric_mapping': record['numeric_mapping'],
                          'runtime_seconds': record['runtime_seconds']})
    result = {'dataset_sha256': fingerprint(rows), 'n': len(rows), 'runs': summaries,
              'macro_f1_delta_b_minus_a': b[1]['macro_f1'] - a[1]['macro_f1'],
              'numeric_mapping_equal': a[0]['numeric_mapping'] == b[0]['numeric_mapping'],
              'connected_validation_clusters': len(clusters), 'bootstrap_seed': 42,
              'bootstrap_draws': 5000 if interval is not None else 0,
              'cluster_bootstrap_percentile_interval_95': interval,
              'limitations': 'Few independent validation groups: interval is descriptive, not reliable significance evidence. Mapping is supervised on training only; official class semantics and premise scope remain unverified. Provided-reference context is not gold/full-booklet evidence. No final test evaluated.'}
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Paired Macro-F1 delta:', result['macro_f1_delta_b_minus_a'],
          'independent connected clusters:', len(clusters))


if __name__ == '__main__':
    main()
