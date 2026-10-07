"""Strict canonical schema. Official field conversion belongs in a reviewed adapter."""
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{number}: invalid JSON") from e
    return rows


def validate_document(doc):
    if not isinstance(doc.get("document_id"), str) or not doc["document_id"]:
        raise ValueError("document_id must be a nonempty string")
    passages = doc.get("passages")
    if not isinstance(passages, list) or not passages:
        raise ValueError("Document must have nonempty passages")
    ids = set()
    for p in passages:
        if not isinstance(p.get("id"), str) or not p["id"] or p["id"] in ids:
            raise ValueError("Passage IDs must be unique nonempty strings")
        ids.add(p["id"])
        if not isinstance(p.get("text"), str) or not p["text"].strip():
            raise ValueError("Passage text must be nonempty")
        if p.get("page") is not None and (type(p["page"]) is not int or p["page"] < 1):
            raise ValueError("Page must be positive integer or null")
    return doc


def load_document(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return validate_document(json.loads(path.read_text(encoding="utf-8")))
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ValueError("PDF input requires installation of the pdf extra") from e
        passages = []
        pdf = PdfReader(path)
        if pdf.is_encrypted:
            raise ValueError("Encrypted PDF input requires a separately supported decryption workflow")
        for page_idx, page in enumerate(pdf.pages):
            for i, text in enumerate(re.split(r"\n\s*\n", page.extract_text(extraction_mode="layout") or "")):
                if text.strip():
                    passages.append({"id": f"page-{page_idx+1}-p-{i+1}", "page": page_idx+1, "text": text.strip()})
        if not passages:
            raise ValueError("PDF has no selectable text; OCR is required and not silently inferred")
    else:
        text = path.read_text(encoding="utf-8")
        passages = [{"id": f"p{i+1}", "page": None, "text": t.strip()} for i, t in enumerate(re.split(r"\n\s*\n", text)) if t.strip()]
    return validate_document({"document_id": path.stem, "passages": passages})


def load_dataset(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty dataset cannot be evaluated")
    seen, documents, groups = set(), {}, {}
    for r in rows:
        for key in ["id", "document_id", "booklet_id", "claim", "claim_language", "document_language"]:
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError(f"Missing or invalid {key}")
        if r["id"] in seen:
            raise ValueError(f"Duplicate example ID {r['id']}")
        seen.add(r["id"])
        if type(r.get("label")) is not int or r["label"] not in (0, 1, 2):
            raise ValueError("Labels must be integer 0/1/2; map official semantics explicitly")
        doc = validate_document({"document_id": r["document_id"], "passages": r.get("passages")})
        fp = json.dumps(doc, sort_keys=True, ensure_ascii=False)
        if r["document_id"] in documents and documents[r["document_id"]] != fp:
            raise ValueError("Inconsistent document content across examples")
        documents[r["document_id"]] = fp
        if r["document_id"] in groups and groups[r["document_id"]] != r["booklet_id"]:
            raise ValueError("Same document assigned to different booklet groups")
        groups[r["document_id"]] = r["booklet_id"]
        gold = r.get("evidence_ids")
        if not isinstance(gold, list) or any(x not in {p["id"] for p in r["passages"]} for x in gold) or len(gold) != len(set(gold)):
            raise ValueError("evidence_ids must be unique IDs from passages (empty allowed)")
    return rows


def fingerprint(rows):
    data = "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in sorted(rows, key=lambda r: r["id"]))
    return hashlib.sha256(data.encode()).hexdigest()


def split_rows(rows, seed=42, validation_fraction=.2, test_fraction=.2):
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError("Require positive validation/test fractions with sum < 1")
    groups = sorted({r["booklet_id"] for r in rows})
    parent = {g:g for g in groups}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    claim_groups = defaultdict(set)
    for r in rows:
        claim_groups[" ".join(words(r["claim"]))].add(r["booklet_id"])
    for gs in claim_groups.values():
        gs = sorted(gs)
        for g in gs[1:]:
            a,b = find(gs[0]),find(g)
            parent[max(a,b)] = min(a,b)
    components = defaultdict(set)
    for g in groups:
        components[find(g)].add(g)
    if len(components) < 3:
        raise ValueError("Need at least three independent booklet/duplicate-claim components")
    weights = Counter(find(r["booklet_id"]) for r in rows)
    ordered = sorted(components,key=lambda g:(-weights[g],hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()))
    targets = {"train":len(rows)*(1-validation_fraction-test_fraction),"validation":len(rows)*validation_fraction,"test":len(rows)*test_fraction}
    assigned_counts = Counter()
    component_assignment = {}
    for i,g in enumerate(ordered):
        empty = [s for s in targets if s not in component_assignment.values()]
        eligible = empty if len(ordered)-i == len(empty) else list(targets)
        selected = max(eligible,key=lambda s:targets[s]-assigned_counts[s])
        component_assignment[g] = selected
        assigned_counts[selected] += weights[g]
    assign = {g:component_assignment[find(g)] for g in groups}
    return {name: [r for r in rows if assign[r["booklet_id"]] == name] for name in ["train", "validation", "test"]}


def inspect_dataset(rows):
    def lengths(xs):
        xs = sorted(xs)
        return {"min": xs[0], "mean": sum(xs)/len(xs), "median": statistics.median(xs), "max": xs[-1], "unit": "unicode_word_count_not_model_tokens"} if xs else None
    duplicate_claims = defaultdict(set)
    for r in rows:
        duplicate_claims[" ".join(words(r["claim"]))].add(r["booklet_id"])
    return {
        "dataset_sha256": fingerprint(rows), "examples": len(rows),
        "documents": len({r["document_id"] for r in rows}), "booklets": len({r["booklet_id"] for r in rows}),
        "fields": sorted(set().union(*(r.keys() for r in rows))),
        "classes": dict(Counter(str(r["label"]) for r in rows)),
        "claim_languages": dict(Counter(r["claim_language"] for r in rows)),
        "document_languages": dict(Counter(r["document_language"] for r in rows)),
        "language_pairs": dict(Counter(f"{r['claim_language']}->{r['document_language']}" for r in rows)),
        "claim_lengths": lengths([len(words(r["claim"])) for r in rows]),
        "evidence_lengths": lengths([sum(len(words(p["text"])) for p in r["passages"] if p["id"] in r["evidence_ids"]) for r in rows]),
        "empty_gold_evidence": sum(not r["evidence_ids"] for r in rows),
        "claims_repeated_across_booklets": sum(len(g)>1 for g in duplicate_claims.values()),
        "leakage_warning": "Use shared booklet_id for all translations, versions and claims of one booklet. Grouping must be checked against official metadata.",
    }
