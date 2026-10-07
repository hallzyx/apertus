"""Train grouped CPU heads and OOF temperature using only frozen training data."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def main():
    import numpy as np
    import sklearn
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from ost_nli.data import fingerprint, load_dataset, words
    from ost_nli.experiments import append_event, utc_now
    from ost_nli.metrics import classification, evaluate
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', required=True)
    p.add_argument('--cap', type=int, choices=[1024, 4096], required=True)
    p.add_argument('--features', choices=['option_logits', 'hidden'], required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--validation', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    cache, out = Path(args.cache), Path(args.output_dir)
    source = json.loads((cache / 'experiment.json').read_text())
    train, val = load_dataset(args.train), load_dataset(args.validation)
    if source['status'] != 'completed' or source['train_sha256'] != fingerprint(train) or source['validation_sha256'] != fingerprint(val):
        raise SystemExit('Completed fingerprint-matched feature cache required')
    for entry in source['files']:
        if hashlib.sha256((cache / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise SystemExit('Feature cache checksum mismatch')
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Existing head run cannot be overwritten')
    matrices = {}
    for name, rows in [('train', train), ('validation', val)]:
        with np.load(cache / str(args.cap) / f'{name}.npz', allow_pickle=False) as values:
            if values['ids'].tolist() != [r['id'] for r in rows] or values['labels'].tolist() != [r['label'] for r in rows]:
                raise SystemExit('Feature row/label alignment failure')
            matrices[name] = values[args.features].astype(np.float64)
        if not np.isfinite(matrices[name]).all():
            raise SystemExit('Nonfinite cache features')
    x, xv = matrices['train'], matrices['validation']
    if args.features == 'option_logits':
        x -= x.mean(axis=1, keepdims=True)
        xv -= xv.mean(axis=1, keepdims=True)
    y = np.asarray([row['label'] for row in train])
    parent = {row['booklet_id']: row['booklet_id'] for row in train}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    seen = {}
    for row in train:
        claim = ' '.join(words(row['claim']))
        if claim in seen:
            parent[find(row['booklet_id'])] = find(seen[claim])
        else:
            seen[claim] = row['booklet_id']
    groups = np.asarray([find(row['booklet_id']) for row in train])
    n_groups = len(set(groups.tolist()))
    if n_groups < 2:
        raise SystemExit('Independent training groups needed for OOF calibration')
    splits = list(GroupKFold(n_splits=min(5, n_groups)).split(x, y, groups))
    for fit, holdout in splits:
        if set(y[fit]) != {0, 1, 2} or set(groups[fit]) & set(groups[holdout]):
            raise SystemExit('Grouped folds need three training classes and isolated documents')
    out.mkdir(parents=True, exist_ok=True)
    registry = Path(__file__).resolve().parents[1] / 'experiments/registry.jsonl'
    record = {'experiment_id': out.name, 'status': 'started', 'timestamp': utc_now(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_cache': cache.name, 'source_cache_git_commit': source['git_commit'],
        'model': source['model'], 'model_revision': source['model_revision'],
        'precision': source['precision'], 'features': args.features, 'n_features': int(x.shape[1]),
        'reference_token_cap': args.cap, 'train_sha256': fingerprint(train), 'validation_sha256': fingerprint(val),
        'dataset_split': 'strict_validation', 'training_configuration': {'n': len(train),
            'seed': 42, 'C_grid': [0.001, 0.01, 0.1, 1.0], 'selection': 'Training grouped OOF Macro-F1 only',
            'groups': n_groups, 'folds': len(splits), 'max_iter': 2000,
            'calibration': 'Scalar temperature minimizing training grouped OOF NLL; no validation labels'},
        'sklearn': sklearn.__version__, 'numpy': np.__version__, 'gpu': 'CPU head; actual Apertus CUDA cache reused',
        'actual_compute_cost': 0.0, 'estimated_compute_cost': 0.0, 'macro_f1': None,
        'notes': 'Frozen real Apertus features; full 902-row training only. Few independent groups limit confidence in calibration and generalization. Latency sums cached GPU encoding and separately measured CPU head; not a live integrated service. Provided references; no gold/full booklet or final test.'}
    append_event(registry, record)
    start = time.perf_counter()
    try:
        def classifier(C):
            return make_pipeline(StandardScaler(), LogisticRegression(C=C, solver='lbfgs',
                                 max_iter=2000, tol=1e-5, random_state=42))
        def softmax(scores, temperature):
            values = scores / temperature
            values -= values.max(axis=1, keepdims=True)
            values = np.exp(values)
            return values / values.sum(axis=1, keepdims=True)
        candidates = []
        for C in record['training_configuration']['C_grid']:
            oof = np.empty((len(train), 3), dtype=np.float64)
            fold_info = []
            for fit, holdout in splits:
                head = classifier(C).fit(x[fit], y[fit])
                if head[-1].n_iter_.max() >= 2000:
                    raise ValueError('Grouped head fit did not converge')
                oof[holdout] = head.decision_function(x[holdout])
                fold_info.append({'fit_n': len(fit), 'held_out_n': len(holdout),
                                  'fit_groups': sorted(set(groups[fit].tolist())),
                                  'held_out_groups': sorted(set(groups[holdout].tolist()))})
            f1 = classification(y.tolist(), np.argmax(oof, axis=1).tolist())['macro_f1']
            candidates.append((f1, C, oof, fold_info))
            print('Training-only grouped head C:', C, 'OOF Macro-F1:', f1, flush=True)
        score, C, oof, fold_info = max(candidates, key=lambda value: (value[0], -value[1]))
        def nll(log_temperature):
            probs = softmax(oof, float(np.exp(log_temperature)))
            return float(-np.log(np.maximum(probs[np.arange(len(y)), y], 1e-15)).mean())
        solution = minimize_scalar(nll, bounds=(float(np.log(0.1)), float(np.log(10))), method='bounded')
        if not solution.success:
            raise ValueError('OOF temperature fit failed')
        temperature = float(np.exp(solution.x))
        head = classifier(C).fit(x, y)
        if head[-1].n_iter_.max() >= 2000:
            raise ValueError('Final head fit did not converge')
        encoded = [json.loads(line) for line in (cache / str(args.cap) / 'validation_rows.jsonl').read_text().splitlines()]
        if [r['id'] for r in encoded] != [r['id'] for r in val]:
            raise ValueError('Encoding metadata row mismatch')
        predictions, raw_predictions, latencies = [], [], []
        for index, row in enumerate(val):
            began = time.perf_counter()
            logits = head.decision_function(xv[index:index+1])
            probability = softmax(logits, temperature)[0]
            raw_probability = softmax(logits, 1.0)[0]
            latency = time.perf_counter() - began
            latencies.append(latency)
            pred = {'id': row['id'], 'label': int(np.argmax(probability)),
                    'probabilities': probability.tolist(), 'context_tokens': encoded[index]['context_tokens'],
                    'latency_seconds': encoded[index]['latency_seconds'] + latency,
                    'head_only_latency_seconds': latency}
            predictions.append(pred)
            raw_predictions.append({**pred, 'probabilities': raw_probability.tolist()})
        metrics, raw_metrics = evaluate(val, predictions), evaluate(val, raw_predictions)
        final = {**record, 'status': 'completed', 'timestamp': utc_now(), 'n': len(val),
                 'selected_C': C, 'temperature': temperature, 'training_oof_macro_f1': score,
                 'runtime_seconds': time.perf_counter() - start, 'macro_f1': metrics['macro_f1'],
                 'per_class_f1': {k: v['f1'] for k, v in metrics['per_class'].items()},
                 'per_language_performance': metrics['by_language'], 'cross_lingual_performance': metrics['cross_lingual'],
                 'calibration_metrics': metrics['calibration'], 'uncalibrated_calibration_metrics': raw_metrics['calibration'],
                 'average_context_tokens': metrics['average_context_tokens'],
                 'average_inference_latency': metrics['average_latency_seconds'],
                 'mean_head_only_latency_seconds': sum(latencies) / len(latencies)}
        scaler, model = head[0], head[1]
        parameters = {'feature_kind': args.features, 'center_option_logits': args.features == 'option_logits',
                      'feature_mean': scaler.mean_.tolist(), 'feature_scale': scaler.scale_.tolist(),
                      'coefficients': model.coef_.tolist(), 'intercept': model.intercept_.tolist(),
                      'class_order': model.classes_.tolist(), 'temperature': temperature,
                      'calibration_source': 'Training-only connected-group out-of-fold scores'}
        audit = {'candidates': [{'C': c, 'oof_macro_f1': f} for f, c, _, _ in candidates],
                 'folds': fold_info, 'oof_temperature_nll_before': nll(0.0),
                 'oof_temperature_nll_after': nll(float(solution.x)),
                 'temperature_fit_uses_validation': False}
        for name, value in [('experiment.json', final), ('metrics.json', metrics), ('head.json', parameters),
                            ('training_audit.json', audit), ('predictions.json',
                             {'dataset_sha256': fingerprint(val), 'configuration': final, 'predictions': predictions})]:
            (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        append_event(registry, final)
        print('Actual frozen Apertus head Macro-F1:', metrics['macro_f1'], 'temperature:', temperature, flush=True)
    except Exception as error:
        failed = {**record, 'status': 'failed', 'timestamp': utc_now(), 'error_type': type(error).__name__}
        (out / 'experiment.json').write_text(json.dumps(failed, indent=2) + '\n')
        append_event(registry, failed)
        raise


if __name__ == '__main__':
    main()
