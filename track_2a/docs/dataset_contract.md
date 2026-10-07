# Dataset contract and leakage controls

The official OST dataset has **not yet been obtained**. This is an internal
canonical interchange format, not a claim about the official field names.
Implement and review the official adapter after inspecting real records and the
official class mapping. Never infer the class mapping from model preferences.

One example per JSONL line:

```json
{"id":"example-001","document_id":"booklet-001-de","booklet_id":"booklet-001","claim":"…","claim_language":"de","document_language":"de","label":0,"evidence_ids":["p1"],"passages":[{"id":"p1","page":1,"text":"…","language":"de"}]}
```

- `booklet_id`: same value for **all translations, versions and claims** from the
  same source voting booklet. This is the split grouping unit.
- `document_id`: unique language/version-specific document identity; contents
  and booklet assignment must be consistent across rows.
- `label`: official integer 0/1/2. Runtime `OST_LABEL_MAP` must be a verified
  bijection from these integers to entailment/contradiction/neutral.
- `evidence_ids`: exact passage IDs, with an empty list allowed when unannotated.
  Empty evidence is excluded from retrieval recall, not treated as perfect recall.
- `page`: one-based original PDF page or null for text without page metadata.
  Do not invent provenance. Preserve original passage segmentation for gold IDs.
- `claim_language` and `document_language`: source metadata, never guessed from
  a document filename. Mixed-language evidence may include passage language.

```bash
PYTHONPATH=src python -m ost_nli inspect data/private/ost.jsonl --output experiments/artifacts/dataset_audit.json
PYTHONPATH=src python -m ost_nli split data/private/ost.jsonl --output-dir data/private/splits --seed 42
```

The split manifest records example IDs, booklet groups, dataset fingerprints and
class/language distributions. Split selection is deterministic and independent
of input row order. It is not stratified: inspect minority-class support and
report absent classes rather than searching split seeds to improve model scores.
Use validation only for model selection. Freeze test before experiments; run the
test split only after final configuration selection. Calibration/head training
must use training groups or a separate training calibration partition.

Additional review required on official data: duplicate booklet identities,
identical texts under different IDs, near-duplicate claims and document editions.
Grouping strings alone cannot prove these are absent. Hash checks detect dataset
changes; they do not certify a correct official adapter.

Token efficiency uses server-reported **total prompt tokens**, including prompt,
evidence and claim. Byte caps are transport/context size controls, not tokenizer
token counts. Missing token usage and unavailable probabilities are null. ECE,
Brier and NLL are reported only for real normalized class probabilities.

Evidence scores match passage IDs, not semantic relevance or span correctness.
Truncated passage text is explicitly marked and needs additional span-level review.
CLI PDF extraction is for text-based PDFs; scanned PDFs fail clearly and require
a documented OCR path before any claim of support for scanned booklets.
