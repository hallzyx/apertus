"""Train-only artifact controls and validation audits; never opens the consumed test."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def connected_groups(rows):
    from ost_nli.data import words
    parent = {r['booklet_id']: r['booklet_id'] for r in rows}
    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    seen = {}
    for r in rows:
        claim = ' '.join(words(r['claim']))
        if claim in seen:
            parent[find(r['booklet_id'])] = find(seen[claim])
        else:
            seen[claim] = r['booklet_id']
    return [find(r['booklet_id']) for r in rows]


def main():
    import numpy as np
    from scipy.optimize import minimize_scalar
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from ost_nli.data import load_dataset, fingerprint, words
    from ost_nli.metrics import evaluate, classification
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='track_2a/data/private/full-booklets-v2')
    p.add_argument('--output', default='track_2a/experiments/apertus-v15-phase2/cpu-audit')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise ValueError('Immutable run already exists')
    out.mkdir(parents=True)
    train, val = [load_dataset(Path(a.data)/f'{s}.jsonl') for s in ('train', 'validation')]
    groups = np.asarray(connected_groups(train))
    folds = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(train, groups=groups))
    y = np.asarray([r['label'] for r in train])
    def probs(z, t=1.):
        z = z/t
        z = np.exp(z-z.max(axis=1, keepdims=True))
        return z/z.sum(axis=1, keepdims=True)
    def inputs(rows, mode):
        if mode != 'metadata':
            return [r['claim'] for r in rows]
        return [{'claim_language':r['claim_language'], 'document_language':r['document_language'],
                 'words':len(words(r['claim'])), 'characters':len(r['claim']),
                 'digits':sum(c.isdigit() for c in r['claim']), 'question':int('?' in r['claim'])} for r in rows]
    results = {}
    for mode in ('claim-word-tfidf', 'claim-char-tfidf', 'metadata'):
        def model(c):
            vector = DictVectorizer() if mode == 'metadata' else TfidfVectorizer(
                analyzer='word' if mode == 'claim-word-tfidf' else 'char',
                ngram_range=(1,2) if mode == 'claim-word-tfidf' else (3,5), min_df=2,
                max_features=30000, sublinear_tf=True)
            return make_pipeline(vector, LogisticRegression(C=c, max_iter=2000, random_state=42))
        xt, xv = inputs(train, mode), inputs(val, mode)
        candidates = []
        for c in (.001, .01, .1, 1.):
            oof = np.empty((len(y),3))
            for fit, held in folds:
                m = model(c).fit([xt[i] for i in fit], y[fit])
                if m[-1].n_iter_.max() >= 2000:
                    raise ValueError('Nonconverged artifact control')
                oof[held] = m.decision_function([xt[i] for i in held])
            score = classification(y.tolist(), oof.argmax(1).tolist())['macro_f1']
            candidates.append((score,c,oof))
        score,c,oof = max(candidates, key=lambda v:(v[0],-v[1]))
        opt = minimize_scalar(lambda lt:float(-np.log(np.maximum(probs(oof,np.exp(lt))[np.arange(len(y)),y],1e-15)).mean()), bounds=(np.log(.1),np.log(10)),method='bounded')
        if not opt.success:
            raise ValueError('Calibration failed')
        t = float(np.exp(opt.x))
        m = model(c).fit(xt,y)
        posterior = probs(m.decision_function(xv),t)
        pred = [{'id':r['id'],'label':int(q.argmax()),'probabilities':q.tolist()} for r,q in zip(val,posterior)]
        metrics = evaluate(val,pred)
        results[mode] = {'metrics':metrics,'C':c,'temperature':t,'train_oof_macro_f1':score,
                         'candidates':[{'C':c,'macro_f1':s} for s,c,_ in candidates]}
        (out/f'{mode}-predictions.json').write_text(json.dumps(pred,indent=2)+'\n')
        print(mode,metrics['macro_f1'],flush=True)
    source = Path('track_2a/experiments/apertus-v15-research-v1/apertus-v15-full-hidden-v1/predictions.json')
    if not source.exists():
        source = Path('track_2a/experiments/apertus-v15-full-hidden-v1/predictions.json')
    original = json.loads(source.read_text())['predictions']
    by_id = {r['id']:r for r in original}
    metrics = evaluate(val, original)
    by_event = {}
    for event in sorted({r['booklet_id'] for r in val}):
        selected = [r for r in val if r['booklet_id']==event]
        by_event[event] = evaluate(selected,[by_id[r['id']] for r in selected])
    buckets = []
    for lo,hi in ((1/3,.5),(.5,.7),(.7,.85),(.85,.95),(.95,1.00000001)):
        selected = [r for r in val if lo <= max(by_id[r['id']]['probabilities']) < hi]
        buckets.append({'lower':lo,'upper':min(hi,1.),'n':len(selected),
                        'accuracy':sum(r['label']==by_id[r['id']]['label'] for r in selected)/len(selected) if selected else None,
                        'mean_confidence':float(np.mean([max(by_id[r['id']]['probabilities']) for r in selected])) if selected else None})
    rng = np.random.default_rng(42)
    events = sorted(by_event)
    samples = []
    for _ in range(2000):
        selected = [r for event in rng.choice(events,len(events),replace=True) for r in val if r['booklet_id']==event]
        samples.append(classification([r['label'] for r in selected],[by_id[r['id']]['label'] for r in selected])['macro_f1'])
    errors = []
    for r in val:
        pred = by_id[r['id']]
        if pred['label'] == r['label']:
            continue
        errors.append({'id':r['id'],'gold':r['label'],'predicted':pred['label'],'claim':r['claim'],
                       'document_language':r['document_language'],'claim_language':r['claim_language'],
                       'booklet_id':r['booklet_id'],'context_method':'full-48000-bytes',
                       'suspected_failure_mode':'undetermined_pending_source_inspection','evidence':[],
                       'observable_claim_features':{'contains_digits':any(c.isdigit() for c in r['claim']),
                                                    'cross_lingual':r['claim_language']!=r['document_language']},
                       'notes':'Error observed on validation; no causal attribution from keyword flags. Source inspection pending.'})
    (out/'error-inventory.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in errors))
    record = {'scope':'train/validation only; no consumed310 input opened',
              'train_sha256':fingerprint(train),'validation_sha256':fingerprint(val),
              'source_prediction_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'artifact_controls':results,'original_hidden_head':metrics,'by_voting_event':by_event,
              'confidence_buckets':buckets,'event_bootstrap':{'n_events':len(events),'draws':2000,'seed':42,
                  'macro_f1_percentile95':np.quantile(samples,[.025,.975]).tolist(),
                  'caution':'Only3 events; exploratory resampling, not reliable population confidence interval; duplicate-connected events not fully independent'},
              'train_connected_groups':dict(Counter(groups.tolist())),
              'validation_connected_groups':dict(Counter(connected_groups(val))),
              'claim_only_apertus_control':'NOT RUN HERE; lexical/metadata probes are supplementary controls',
              'compute_cost_usd':0}
    (out/'audit.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
