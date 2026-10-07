"""Transparent lexical baseline; no claim of cross-lingual semantic matching."""
import math
from collections import Counter
from .data import words


def retrieve(passages, claim, k=5, diversify=False, k1=1.5, b=.75):
    if k < 1:
        raise ValueError("k must be positive")
    counts = [Counter(words(p["text"])) for p in passages]
    average = sum(sum(c.values()) for c in counts)/len(counts)
    query = set(words(claim))
    df = {w: sum(w in c for c in counts) for w in query}
    ranked = []
    for i, (p, c) in enumerate(zip(passages, counts)):
        length = sum(c.values())
        score = sum(math.log(1+(len(passages)-df[w]+.5)/(df[w]+.5))*c[w]*(k1+1)/(c[w]+k1*(1-b+b*length/max(average, 1))) for w in query if c[w])
        ranked.append((score, i, p))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    if not diversify:
        return [{**p, "retrieval_score": s} for s, _, p in ranked[:k]]
    selected = []
    while ranked and len(selected)<k:
        def utility(item):
            toks = set(words(item[2]["text"]))
            redundancy = max((len(toks & set(words(p["text"]))) / max(1, len(toks | set(words(p["text"])))) for p in selected), default=0)
            maxscore = max(x[0] for x in ranked) or 1
            return .75*item[0]/maxscore-.25*redundancy
        best = max(range(len(ranked)), key=lambda j: (utility(ranked[j]), -ranked[j][1]))
        s, _, p = ranked.pop(best)
        selected.append({**p, "retrieval_score": s})
    return selected
