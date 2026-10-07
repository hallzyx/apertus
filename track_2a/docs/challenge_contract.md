# Adopted OST Track 2A contract

Source: the user's detailed challenge contract supplied in this conversation.
This document records implementation requirements, not a claim of independent
verification of organizer material.

- Inference input: complete official voting booklet + natural-language claim.
- Fixed labels: 0 entailment, 1 neutral, 2 contradiction.
- Primary inference model: Apertus v1.5. Supporting retrieval models are disclosed.
- Support DE/FR/IT and all nine document–claim language pairs.
- Return source passages with page, exact extracted text, paragraph ID and PDF hash.
- Primary score: three-class Macro-F1. Also report language/cross-language slices,
  actual context tokens, runtime, and clearly identified evidence diagnostics.
- `reference_string` is a development/oracle diagnostic, never production input.
- CLI signature, evidence schema, hardware, time and Internet availability are
  unspecified; choose and document a deterministic interface without blocking.
- Clean-checkout `make run` must start Docker; preserve `track_2a/` structure.
- Large PDFs, model weights, downloaded datasets and binary feature caches are
  downloaded/generated separately, not committed as application data.
- Internal 310-row split is our held-out development comparison, not organizer
  hidden evaluation. Do not train on its labels or repeatedly tune against it.
- Submission: http://hackapertus.ch/online-hack/submissions . Devpost is unnecessary.
- Model predictions are document-grounded relations, not real-world truth or advice.

References supplied: https://github.com/HackApertus/project-template/tree/main/track_2a
https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets
https://www.bk.admin.ch/de/sammlung-der-abstimmungsbuechlein-seit-1978

Remaining external prerequisite observed: official swiss-ai/Apertus-v1.5-8B
requires approved Hugging Face access and authentication (HTTP 401 without it).
No CLI/evidence-format clarification or Devpost access is required.
