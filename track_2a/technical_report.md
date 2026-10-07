# Technical report — Apertus Evidence Lab

Track 2A · OST · Hack Apertus Online. **Research checkpoint, not final submission.**

## 1. Problem and current result

The goal is multilingual three-class NLI grounded in Swiss voting booklets, with
relevant source evidence and efficient Apertus inference. The official data adapter,
evaluation pipeline, leakage-aware partitions and Docker application are implemented.
No Apertus inference run has executed yet. No claim of improved Macro-F1, calibrated
Apertus confidence or final challenge readiness is made.

A CPU training-majority statistical floor scored **0.158379 Macro-F1** on 276 strict
validation examples. This validates evaluator execution on real data; it is not an
Apertus model or a submission architecture. The full final test partition is unscored.

## 2. Architecture and Apertus usage

Input JSON/text/PDF → passage extraction with provenance → candidate context selector
→ Apertus OpenAI-compatible inference endpoint → strict JSON decision → independent
metric evaluation and append-only experiment registry.

Implemented candidates: provided-reference context, bounded full context for actual
full documents, true gold evidence when annotated, BM25 and optional lexical
diversification. Prompted and JSON-schema-constrained output paths exist. Evidence
and claim are sent as data, with multilingual semantic/negation/numerical instructions.
Invalid or truncated generation fails rather than being converted to a default class.
Explanation generation is not required for classification.

No context method has been selected empirically as the final architecture. Dense
retrieval, hybrid retrieval, decomposition, direct class-logit scoring, frozen heads,
QLoRA and adaptive computation remain unrun hypotheses. They will be added only after
baseline comparisons demonstrate a useful reason.

Public fallback model identified: `swiss-ai/Apertus-8B-Instruct-2509`, revision
`b946d40447b2b597999b9c86d44bee0b452c919f`, Apache-2.0. Its model card requires
Transformers >=4.56.0; config specifies 32 layers, hidden dimension 4096,
8 KV heads and a 65,536-position maximum. Model weights have not been downloaded.
`Apertus-v1.5-8B` was also inspected, but config access without authentication
returned HTTP 401. The public 8B fallback avoids making a new gated credential
an unconditional prerequisite. Exact serving support, quantization and memory must
be verified before renting a GPU. Never assume the published maximum context fits
24 GB VRAM; use the actual tokenizer and measured KV/weight/runtime memory.

## 3. Official dataset and methodology

Source: [OSTswiss/MNLIoverSwissVotingBooklets](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets).
Pinned revision: `9ff08597fb79dc68cbb3af9eb1388f34d21223e6`.
JSONL SHA-256: `87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd`.
Source metadata declares MIT. The complete card currently contains only license
metadata; it does not define numeric class semantics or official scoring protocol.

| Property | Observed value |
|---|---:|
| Examples | 1,488 |
| Voting dates / source booklet groups | 20 |
| Language-specific PDF URLs | 60 |
| Distinct provided reference strings | 667 |
| Classes 0 / 1 / 2 | 495 / 498 / 495 |
| Claims DE / FR / IT | 495 / 497 / 496 |
| References DE / FR / IT | 455 / 569 / 464 |
| Exact duplicate claim/reference/language/label examples | 335 |
| Normalized claims repeated across dates | 12 |
| Claim word count: min / median / max | 6 / 21 / 63 |

Word counts use Unicode word segmentation, **not model tokens**. All nine language
pairs occur. Counts by pair are in `experiments/dataset_audit.json`.

Fields observed: claim, claim_language, reference_string, entailment_label,
baseline_score, reference_language, booklet_publish_date, booklet_download_date,
booklet_url, vote. `baseline_score` is excluded from inputs/features and evaluation;
its origin and meaning are undocumented, and using it risks label leakage.

The canonical adapter retains the provided premise, source URL and original row
index. Reference paragraphs have no fabricated page numbers. `evidence_ids=[]`
and `evidence_annotation_available=false` explicitly mark missing minimal-gold
annotations. Provided-reference paragraphs are not called gold evidence. Their
word length is not represented as an observed gold-evidence length.

The source's neutral examples may concern claims about other referendum proposals.
Expanding from the given reference to a full booklet could change the intended
relation. The official task must clarify the admissible premise scope before
interpreting full-context versus reference-context scores as retrieval improvement.

## 4. Leakage-aware frozen partitions

First group all language variants by booklet publication date. Connect date groups
sharing a normalized identical claim. This produces seven connected components.
Assign components by descending size to training/validation/test using target count
deficits, with seed-42 hash tie breaks. No label-based seed or partition search.

| Partition | Examples | Dates | Classes 0 / 1 / 2 |
|---|---:|---:|---|
| Train | 902 | 13 | 317 / 296 / 289 |
| Validation | 276 | 3 | 86 / 98 / 92 |
| Test | 310 | 4 | 92 / 104 / 114 |

The full manifest, row IDs, fingerprints and distributions are committed at
`experiments/split_manifest.json`. Dates and normalized identical claims do not
cross partitions. Translated/paraphrased source claims and document editions still
need review: these controls do not prove zero semantic leakage. Exact duplicates
are retained to preserve the source population; report deduplicated sensitivity
results before claiming robust model improvements. With only three validation
dates, uncertainty must be assessed at booklet level, not by treating rows as
independent observations.

## 5. Evaluation and measured floor

Macro-F1 is the arithmetic mean of F1 for fixed classes 0/1/2, with absent/undefined
class F1 set to zero. The evaluator reports per-class support, confusion matrices,
claim-language and language-pair slices, and cross-lingual subsets. Exact ID matching
and dataset fingerprints are required. Empty runs cannot pass.

Prompt-token usage is the inference server's observed prompt-token count, including
system prompt, evidence and claim. Missing usage remains null; byte caps are not
token counts. Latency measures context selection plus endpoint request/parsing.
Evidence metrics require actual annotated passage IDs and otherwise remain null.
ECE (10 equal-width bins), multiclass Brier score and NLL require normalized real
probabilities. Generated labels do not manufacture probabilities.

| Setup | Validation Macro-F1 | Status |
|---|---:|---|
| Training-majority CPU floor | 0.158379 | Executed; not Apertus; no semantic inference |
| A: maximal practical full booklet → Apertus 8B | — | Unrun: endpoint and full PDFs unavailable; premise scope needs confirmation |
| B: true gold evidence → Apertus 8B | — | Unrun: gold annotation not provided in current source |
| Provided reference → Apertus 8B | — | Unrun: secure GPU/endpoint and verified class mapping needed |
| BM25 → Apertus 8B | — | Implemented candidate, not evaluated |
| Frozen representations / QLoRA / adaptive inference | — | No GO evidence; not started |

The floor predicts class 0 because it is the training majority, using training-only
class priors. Validation accuracy is 0.311594; per-class F1 is
0.475138 / 0 / 0. Per-language Macro-F1: DE 0.146032, FR 0.183575,
IT 0.140056. Cross-lingual Macro-F1 is 0.153846 on 190 examples.
Prior calibration: ECE 0.039847, Brier 0.668201, NLL 1.100875.
This weak constant predictor illustrates why low ECE alone does not establish
useful decisions. It consumes no LLM tokens and its CPU assignment timing is not
an Apertus latency comparison.

Artifacts and full metrics: `experiments/cpu-majority-v1/`. Experiment code commit:
`3b203db` (full SHA in experiment record). Registry includes started/completed events;
failed real model runs also retain their failure status and partial predictions.
There are no unreported successful Apertus runs.

## 6. Compute budget and lifecycle

Hard Vast budget USD 10; task-estimated spend USD 0; actual account spend unknown.
No paid instance has been created by this task. Account-wide instance absence is
not verified because `VAST_API_KEY` has no saved secure binding and is absent in
the runtime. The chat-supplied credential was not stored or used; replace it using
the secure environment settings.

`experiments/budget.json` tracks totals, task instance IDs and a USD 2 reserve.
The `budget-plan` command checks finite costs, runtime, disk, transfer, a margin
and remaining research budget before any rental. It does not create an instance.
Current offers, balance, actual storage charges and recovery margins must be
verified through Vast before spending. Prefer a sufficient 4090/5090-class offer;
no H100/A100/70B cost is justified by current evidence.

For any future task-created instance: record its ID, GPU, price and purpose;
set a bounded experiment; transfer and verify results in durable storage; destroy
the instance; verify its absence through the API. Do not touch unrelated user
instances. Never treat stopping as destruction or leave GPU idle during CPU work.

## 7. Reproducibility and software validation

The template layout is retained. `make run` starts the Docker workbench; image
base and PDF package are pinned, with dependency hash and TLS verification enabled.
The cloud helper resolves runtime proxy DNS and mounts trusted CA configuration,
without committing secrets or hard-coded proxy addresses. Tested install_script and
start_skill contents, network additions and the missing secure Vast requirement
are saved in the environment draft. They are not proof of publication or fresh-task
restoration.

Validation completed so far: 36 host tests, including PDF extraction/provenance,
grouped split and duplicate-claim isolation, metric fixtures, schema-constrained
request construction, actual HTTP client/evaluator round trip with a **synthetic
mock endpoint**, and rejection of stale predictions. All 36 tests also passed inside Docker with no skips. Clean-checkout `make run`,
DE/FR/IT retrieval and original-page provenance passed after a restart using current
code. Evidence is recorded in `experiments/software_validation.json`. Task-created
local test containers were removed after validation.

These are software checks, not evidence that real Apertus inference or official
challenge accuracy works. The Docker UI's health response explicitly distinguishes
configuration presence from model connectivity and benchmark validation.

## 8. Real Apertus CPU experiments

Original `swiss-ai/Apertus-8B-Instruct-2509`, revision
`b946d40447b2b597999b9c86d44bee0b452c919f`, has now executed on this
machine. All four original BF16 weight shards were checked against official LFS
SHA-256 metadata. The CPU stack is Torch 2.8.0+cpu / Transformers 4.56.2,
SDPA, four threads, seed 42, no remote model code. Dependencies are hash locked
in `requirements-cpu.lock`; `scripts/download_cpu_model.py` reproduces download
and checksum verification including the official chat template.

The completed three-case cross-language synthetic diagnostic achieved **2/3**
for both greedy generation and constrained A/B/C first-token choice. It failed
the FR-premise / IT-claim contradiction case by returning neutral. Mean prompt
length was 130 tokens and measured mean inference latency was **3.663 seconds**.
These are real Apertus outputs, but synthetic diagnostic accuracy is not an OST
score. Full artifacts and candid script/checkout provenance are in
`experiments/apertus-8b-cpu-smoke-v1/`. Vast rental expenditure is zero.

A real OST CPU run was launched from commit `a4091d0` against the frozen strict
validation partition of 276 rows. A correspondence between semantic options and
numeric labels is fitted exclusively on 30 balanced training examples; this is
supervision and does not verify the official class-name convention. The prompt
uses the supplied reference, capped to its first 1,024 tokens, not a complete
booklet or annotated gold evidence. All truncation and raw first-token logits
are retained; no calibrated probabilities are asserted. Long-reference prefill
has measured roughly 15–40 seconds per training example, substantially slower
than the short smoke. The environment restart interrupted this run after 30 training and 19 validation
examples; partial artifacts are preserved and its status is **interrupted**;
there is no completed validation Macro-F1 or final-test result to report. Inspect
`experiments/apertus-8b-cpu-reference-1024-v1/experiment.json` for the actual
state, and do not score an unfinished run as if it covered all validation rows.

## 9. Blockers, limitations and next research decision

Secure Vast authentication is now verified (HTTP 200). The account initially
reported USD 10 credit. A task-created RTX 4090 (instance 54573150, pinned CUDA
image, USD 0.40233/hour including 45 GB storage) was provisioned for paired
1,024/4,096-token reference baselines. Its last observed state was GPU preparation;
no GPU inference was verified. S3 logs were rejected by the runtime proxy and SSH
transport was unavailable. The attempt was aborted, its metadata pushed, and the
instance destroyed; API v1 confirmed an empty account instance list. See
`experiments/vast-provision-v1/` and `budget.json` for timestamped cleanup and credit
observations. Required `s3.amazonaws.com` and `ssh8.vast.ai` destinations are saved
in the environment draft but require user application before another rental. The public original Apertus 8B model has been downloaded with verified official
weight hashes and executed on CPU; no GPU or serving endpoint is prepared.
The official event guide/terms URLs return HTTP 403 (error 1010). Multiple official
booklet PDF URLs were probed and returned HTTP 503 upstream connection failures.
The dataset has no minimal gold-evidence labels or exact page provenance.

Required next actions: apply the saved log/SSH network destinations; supply/access
the full OST specification, class mapping and official evaluator requirements;
confirm whether evaluation is against supplied reference excerpts or complete
booklets and whether gold evidence exists. Then cost and run the cheapest paired
8B baseline. No training, representation-head or fine-tuning GO decision is
supported until those baseline results exist.

The CLI and workbench are research scaffolding, not a completed challenge-compliant
submission. PDF OCR, official interface/schema compliance, completed OST Apertus metrics,
final system selection, final report, quantitative demo comparison and account-wide
billable-resource verification remain outstanding.

## References and licensing

- [Official template](https://github.com/HackApertus/project-template), revision `7f2382275461baf3fa6c8855d157d86abffe9f0e`.
- [OST source dataset](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets), revision recorded above.
- [Public Apertus 8B model](https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509), revision recorded above; model card/config cached in docs.
- [Official challenge guide](https://hackapertus.notion.site/getting-started-guide-onlinehack), inaccessible in this runtime.

Code Apache-2.0; this report CC-BY-4.0 following the template. OST source metadata
MIT; pypdf BSD-3-Clause. Event-specific terms must be checked before submission.
