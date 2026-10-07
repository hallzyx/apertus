import math
import statistics


def classification(y, predicted):
    if not y or len(y) != len(predicted):
        raise ValueError("Nonempty aligned labels and predictions required")
    if any(type(x) is not int or x not in (0, 1, 2) for x in y+predicted):
        raise ValueError("Labels must be 0/1/2")
    matrix = [[0]*3 for _ in range(3)]
    for truth, pred in zip(y, predicted):
        matrix[truth][pred] += 1
    scores = {}
    for c in range(3):
        tp = matrix[c][c]
        fp = sum(matrix[i][c] for i in range(3))-tp
        fn = sum(matrix[c])-tp
        scores[str(c)] = {"precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else 0., "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0., "support": sum(matrix[c])}
    return {"n": len(y), "macro_f1": sum(v["f1"] for v in scores.values())/3, "accuracy": sum(matrix[c][c] for c in range(3))/len(y), "per_class": scores, "confusion_matrix": matrix, "class_order": [0, 1, 2]}


def calibration(y, probabilities, bins=10):
    if len(y) != len(probabilities) or not y:
        raise ValueError("Nonempty aligned probabilities required")
    if bins < 1:
        raise ValueError("bins must be positive")
    for p in probabilities:
        if len(p) != 3 or any(not math.isfinite(x) or x < 0 or x > 1 for x in p) or abs(sum(p)-1)>1e-6:
            raise ValueError("Require normalized finite three-class probabilities")
    brier = sum(sum((p[c]-(c==t))**2 for c in range(3)) for t, p in zip(y, probabilities))/len(y)
    nll = -sum(math.log(max(1e-15, p[t])) for t, p in zip(y, probabilities))/len(y)
    groups = [[] for _ in range(bins)]
    for t, p in zip(y, probabilities):
        pred = max(range(3), key=lambda c:p[c])
        confidence = p[pred]
        groups[min(bins-1, int(confidence*bins))].append((confidence, pred==t))
    ece = sum(abs(sum(x[0] for x in g)/len(g)-sum(x[1] for x in g)/len(g))*len(g)/len(y) for g in groups if g)
    return {"ece": ece, "ece_bins": bins, "brier_score": brier, "negative_log_likelihood": nll}


def evaluate(rows, predictions):
    ids = [p.get("id") for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in rows}:
        raise ValueError("Predictions must match dataset IDs exactly, without duplicates")
    by_id = {p["id"]:p for p in predictions}
    def subset(selected):
        return classification([r["label"] for r in selected], [by_id[r["id"]]["label"] for r in selected]) if selected else None
    result = subset(rows)
    result["by_language"] = {lang:subset([r for r in rows if r["claim_language"] == lang]) for lang in sorted({r["claim_language"] for r in rows})}
    result["by_language_pair"] = {pair:subset([r for r in rows if (r["claim_language"],r["document_language"]) == pair]) for pair in sorted({(r["claim_language"],r["document_language"]) for r in rows})}
    result["by_language_pair"] = {f"{a}->{b}":v for (a,b),v in result["by_language_pair"].items()}
    result["cross_lingual"] = subset([r for r in rows if r["claim_language"] != r["document_language"]])
    aligned = [by_id[r["id"]] for r in rows]
    for name in ["context_tokens", "latency_seconds"]:
        observed = [p[name] for p in aligned if p.get(name) is not None]
        if any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in observed):
            raise ValueError(f"Invalid {name}")
        result[f"average_{name}"] = statistics.mean(observed) if len(observed)==len(rows) else None
        result[f"{name}_coverage"] = len(observed)/len(rows)
    probs = [p.get("probabilities") for p in aligned]
    result["calibration"] = calibration([r["label"] for r in rows], probs) if all(p is not None for p in probs) else None
    result["probability_coverage"] = sum(p is not None for p in probs)/len(rows)
    evidence = []
    for r, p in zip(rows, aligned):
        gold = set(r["evidence_ids"])
        if gold and "evidence_ids" in p:
            retrieved = set(p["evidence_ids"])
            hits = len(gold & retrieved)
            evidence.append((hits/len(gold), hits/len(retrieved) if retrieved else 0., int(gold <= retrieved)))
    result["evidence"] = {"n_annotated":len(evidence), "recall":statistics.mean(x[0] for x in evidence) if evidence else None, "precision":statistics.mean(x[1] for x in evidence) if evidence else None, "all_gold_retrieved":statistics.mean(x[2] for x in evidence) if evidence else None, "definition":"Passage-ID match; excludes empty gold evidence; not semantic or span correctness"}
    return result
