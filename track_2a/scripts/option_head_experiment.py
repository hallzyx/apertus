"""Cheap training-only linear head over real frozen Apertus option logits."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from ost_nli.data import fingerprint, load_dataset, words
    from ost_nli.experiments import append_event, utc_now
    from ost_nli.metrics import evaluate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    source, out = Path(args.source_run), Path(args.output_dir)
    record = json.loads((source / 'experiment.json').read_text())
    bundle = json.loads((source / 'predictions.json').read_text())
    train, val = load_dataset(args.train), load_dataset(args.validation)
    if record['status'] != 'completed' or bundle['dataset_sha256'] != fingerprint(val):
        raise SystemExit('Complete source run with matching frozen validation required')
    if record['training_dataset_sha256'] != fingerprint(train):
        raise SystemExit('Training fingerprint changed')
    if {' '.join(words(r['claim'])) for r in train} & {' '.join(words(r['claim'])) for r in val}:
        raise SystemExit('Claim leakage')
    if {r['booklet_id'] for r in train} & {r['booklet_id'] for r in val}:
        raise SystemExit('Booklet leakage')
    scored = [json.loads(line) for line in (source / 'training_scores.jsonl').read_text().splitlines()]
    training_by_id = {row['id']: row for row in train}
    if len({row['id'] for row in scored}) != len(scored) or {row['id'] for row in scored} != set(record['training_mapping_ids']):
        raise SystemExit('Training score IDs are not the frozen training sample')
    if any(row['id'] not in training_by_id or row['label'] != training_by_id[row['id']]['label'] for row in scored):
        raise SystemExit('Head must fit verified training labels only')
    # Reject incomplete/stale predictions before fitting or scoring another model.
    evaluate(val, bundle['predictions'])
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Existing experiment output cannot be overwritten')
    def features(rows):
        values = np.asarray([row['first_token_option_logits'] for row in rows], dtype=np.float64)
        if values.shape != (len(rows), 3) or not np.isfinite(values).all():
            raise ValueError('Three actual finite Apertus logits required')
        return values - values.mean(axis=1, keepdims=True)
    x, y = features(scored), np.asarray([row['label'] for row in scored])
    if set(y) != {0, 1, 2}:
        raise SystemExit('Training sample lacks a class')
    out.mkdir(parents=True, exist_ok=True)
    registry = Path(__file__).resolve().parents[1] / 'experiments/registry.jsonl'
    configuration = {'experiment_id': out.name, 'status': 'started', 'timestamp': utc_now(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model': record['model'], 'model_revision': record['model_revision'],
        'precision': record['precision'], 'source_experiment_id': record['experiment_id'],
        'source_predictions_sha256': hashlib.sha256((source / 'predictions.json').read_bytes()).hexdigest(),
        'training_dataset_sha256': fingerprint(train), 'validation_dataset_sha256': fingerprint(val),
        'dataset_split': 'strict_validation', 'training_configuration': {'n': len(scored),
            'features': 'Three centered frozen Apertus next-token option logits; no hidden-state cache',
            'method': 'StandardScaler fitted on training only + L2 multinomial LogisticRegression',
            'C': 1.0, 'solver': 'lbfgs', 'max_iter': 1000, 'seed': 42,
            'hyperparameters_selected_without_validation': True,
            'training_ids': [row['id'] for row in scored]},
        'sklearn': sklearn.__version__, 'numpy': np.__version__, 'gpu': 'CPU head; source encoding on RTX 4090',
        'estimated_compute_cost': 0.0, 'actual_compute_cost': 0.0, 'macro_f1': None,
        'notes': 'Reuses actual GPU logits; no new rental cost. Raw trained classifier probabilities, not independently calibrated. Training sample is small and concentrated in two connected groups; grouped calibration was not performed. Latency sums measured source encoding and per-row CPU head inference, not a live integrated service. Official class semantics/full-booklet scope unverified; no final test.'}
    append_event(registry, configuration)
    started = time.perf_counter()
    head = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver='lbfgs',
                         max_iter=1000, tol=1e-6, random_state=42))
    try:
        head.fit(x, y)
        if head[-1].n_iter_.max() >= 1000:
            raise ValueError('Head solver did not converge; do not publish a score')
    except Exception as error:
        failed = {**configuration, 'status': 'failed', 'timestamp': utc_now(),
                  'error_type': type(error).__name__}
        (out / 'experiment.json').write_text(json.dumps(failed, indent=2) + '\n')
        append_event(registry, failed)
        raise
    val_by_id = {row['id']: row for row in bundle['predictions']}
    predictions, head_latencies = [], []
    for row in val:
        encoded = val_by_id[row['id']]
        begin = time.perf_counter()
        probabilities = head.predict_proba(features([encoded]))[0]
        latency = time.perf_counter() - begin
        head_latencies.append(latency)
        pred = {**encoded, 'label': int(head[-1].classes_[int(np.argmax(probabilities))]),
                'probabilities': probabilities.tolist(),
                'probability_note': 'Uncalibrated trained multinomial classifier posterior; no validation labels used in fitting',
                'latency_seconds': encoded['latency_seconds'] + latency,
                'head_only_latency_seconds': latency}
        predictions.append(pred)
    metrics = evaluate(val, predictions)
    final = {**configuration, 'timestamp': utc_now(), 'status': 'completed', 'n': len(val),
             'runtime_seconds': time.perf_counter() - started, 'macro_f1': metrics['macro_f1'],
             'per_class_f1': {k: v['f1'] for k, v in metrics['per_class'].items()},
             'per_language_performance': metrics['by_language'], 'cross_lingual_performance': metrics['cross_lingual'],
             'evidence_metrics': metrics['evidence'], 'calibration_metrics': metrics['calibration'],
             'average_context_tokens': metrics['average_context_tokens'],
             'average_inference_latency': metrics['average_latency_seconds'],
             'mean_head_only_latency_seconds': sum(head_latencies) / len(head_latencies),
             'macro_f1_delta_from_source': metrics['macro_f1'] - record['macro_f1']}
    scaler, classifier = head[0], head[1]
    parameters = {'feature_definition': 'Subtract row mean from three option logits',
                  'feature_mean': scaler.mean_.tolist(), 'feature_scale': scaler.scale_.tolist(),
                  'coefficients': classifier.coef_.tolist(), 'intercept': classifier.intercept_.tolist(),
                  'class_order': classifier.classes_.tolist(), 'calibrated': False}
    for name, value in [('experiment.json', final), ('metrics.json', metrics), ('head.json', parameters),
                        ('predictions.json', {'dataset_sha256': fingerprint(val), 'configuration': final,
                                              'predictions': predictions})]:
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    append_event(registry, final)
    print('Real Apertus option-head Macro-F1:', metrics['macro_f1'],
          'delta:', final['macro_f1_delta_from_source'])


if __name__ == '__main__':
    main()
