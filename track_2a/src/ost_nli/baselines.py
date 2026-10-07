"""CPU-only statistical floor, never a submitted Apertus system."""
import time
from collections import Counter
from .data import words


def training_majority(train, validation):
    if {r["booklet_id"] for r in train} & {r["booklet_id"] for r in validation}:
        raise ValueError("Training and validation share booklet groups")
    if {" ".join(words(r["claim"])) for r in train} & {" ".join(words(r["claim"])) for r in validation}:
        raise ValueError("Training and validation share normalized identical claims")
    counts = Counter(r["label"] for r in train)
    probabilities = [counts[c]/len(train) for c in range(3)]
    majority = max(range(3),key=lambda c:probabilities[c])
    predictions = []
    for r in validation:
        start = time.perf_counter()
        pred = {"id":r["id"],"label":majority,"probabilities":probabilities.copy(),"context_tokens":0,"model":"training-majority-prior","latency_seconds":time.perf_counter()-start}
        predictions.append(pred)
    return predictions
