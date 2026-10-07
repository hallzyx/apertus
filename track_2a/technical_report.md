# Apertus Evidence Lab — Track 2A OST technical report

## 1. Summary

The adopted challenge contract requires booklet + claim, fixed labels 0 entailment /
1 neutral / 2 contradiction, Apertus v1.5, and transparent source passages. Production
must not require reference strings. The app now implements PDF processing,
multilingual semantic/BM25 fusion retrieval, source provenance, a Docker UI and
single/batch CLI. Exact CLI/evidence schemas are project choices, not blockers.

**Completed real Apertus v1.5 experiments:** ten validation baselines, two training-only decision heads, and the frozen 310-row internal holdout. On 276 validation rows, the best base configuration reaches Macro-F1 **0.671612**, the option-logit classifier **0.721862**, and the selected hidden classifier **0.894461**. The selected system reaches **0.899920** on 310 internal holdout examples. These are internal, booklet-disjoint results, not the organizer's hidden score. No matched base-model NLI run on the 310 holdout was made, so the 22.3-point improvement is a validation comparison only.

Apertus weights remain frozen: no LoRA or model-weight fine-tuning. A standardized multinomial logistic-regression head was fitted on 902 training examples. Model files, source predictions, head parameters and all recovered artifacts are SHA-256 verified. Validation/features were encoded on an A40; the final heldout evaluation and three actual multilingual PDF CLI cases ran on an RTX A6000, using BF16 text weights, FP32 tokenizer submodules, no quantization, Torch 2.8.0/CUDA 12.8 and the pinned official fork.

A mistaken controller reboot interrupted the heldout evaluation after 26 persisted predictions. Recovered output preserves those exact bytes and appends only the 284 missing examples with unchanged selection, head and inputs. No partial-test score was used for selection; no training or calibration followed test inference. The resume audit records one logical fixed holdout, not a claim of exactly 310 GPU calls: an interrupted in-flight request may have been retried, and three public CLI integration requests were made separately. Temporary Hugging Face credentials were removed from Vast; cached inference needs no token. Earlier Apertus 2509 reference-only results remain historical research.

## 2. Architecture

PDF → per-page text → overlapping source passages (1,200 characters, overlap 200)
→ validation-selected full/capped context (48,000 document-context bytes)
→ configured Apertus v1.5 endpoint → strictly parsed numeric relation → evidence
and runtime/token metadata. BM25, E5 dense and hybrid k=5 retrieval remain optional
comparison modes; they are not the selected production context. Passage text remains an exact substring of extracted
page text, with page/offsets and original PDF SHA-256. Production uses no labels,
reference strings or dataset baseline scores.

The first extraction used PDF layout mode and fragmented words. Five training-only
extraction comparisons motivated plain extraction; mean full-booklet reference
5-gram coverage improved from 0.2834 to 0.8280 on validation. Both measurements are
retained; matching is approximate and not organizer evidence grading. OCR is not
implemented. Full context is explicitly byte-capped/truncation-reported when used. All 276 validation full-context inputs were truncated at the 48,000-byte document-context budget; this baseline is full/capped, not an uncapped complete-booklet experiment.

## 3. Use of Apertus

Required model: `swiss-ai/Apertus-v1.5-8B`, pinned metadata revision
`a411d838600baf0e3635a3daf66fb7c55fc97bb6`, architecture
`Apertus1p5ForConditionalGeneration`, model type `apertus1p5`. It is distinct from
legacy Apertus 2509; old hidden-head parameters cannot be transferred unchanged.
The selected native endpoint runs the frozen hidden classifier (deployment/head-v15.json); the frontend uses an authorized v1.5 endpoint through
LLM_NAME/LLM_BASE_URL/LLM_API_KEY. Model names must identify the required generation;
The native runner uses the official Transformers fork at
`3797303dda74844e3d1f8977ff5518bb91f818b4`. Text attention uses SDPA while
vision/audio submodels use their supported eager implementation. CPU verification
used Torch 2.8.0+cpu and BF16 text weights, with no quantization; see
`experiments/v15-access-v1/real_cpu_preflight.json`. Cached weights need no token.
The temporary task credential was deleted from Vast after checksum verification.

Supporting model: `intfloat/multilingual-e5-small`, immutable revision
`614241f622f53c4eeff9890bdc4f31cfecc418b3`, MIT, CPU, original FP32 weights with
source LFS SHA-256 verification. Masked mean pooling + L2 normalization, `query:` /
`passage:` prefixes, 512-token embedding cap, 384 dimensions. It performs retrieval,
never the NLI decision. No substitute central NLI model is used.

## 4. Data

OSTswiss/MNLIoverSwissVotingBooklets, revision
`9ff08597fb79dc68cbb3af9eb1388f34d21223e6`, original JSONL SHA-256
`87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd`.
1,488 examples, DE/FR/IT, all nine document–claim language pairs. All 60 linked
source PDFs downloaded and parsed; no rows missing from 902 train / 276 validation /
310 internal holdout. PDF manifests record original URLs, hashes, pages and byte sizes.
Raw PDFs/data/weights are separately downloaded and ignored by Git.

Partitions group voting dates and connect dates sharing normalized identical claims,
without label-based partition search. Only two connected train components and two
validation components remain; this limits generalization/calibration confidence.
Reference strings are retained exclusively for oracle/alignment diagnostics.
The 310 holdout is internal, not the organizer's hidden evaluation.

## 5. Evaluation

Three-class Macro-F1 and per-class/language/pair/cross-language metrics are implemented;
actual endpoint token usage and time are recorded. Probabilities/calibration are null
when the endpoint supplies only a hard class. No fake posterior or token estimate.

Real retrieval-only validation on 276 full-booklet inputs, k=5:

| Retrieval | Mean reference 5-gram coverage | Cross-language coverage |
|---|---:|---:|
| BM25 | 0.147356 | 0.080880 |
| Multilingual E5 dense | 0.200044 | 0.191805 |
| Hybrid | 0.213526 | 0.187803 |

Hybrid k=5 was frozen from validation before one internal holdout retrieval comparison.
On the 310-row internal holdout, BM25/dense/hybrid mean reference overlap is
0.194468/0.201401/0.253554; cross-language overlap is
0.116697/0.176814/0.203536. The final NLI holdout result is reported separately below.

This diagnostic measures overlap with available references, not semantic evidence
correctness, official evidence score, or NLI Macro-F1. Dense slightly exceeds hybrid
on cross-language overlap; hybrid wins aggregate overlap. NLI superiority is unproven.

| Validation setup | Macro-F1 | Valid outputs | Context tokens | Latency | Evidence |
|---|---:|---:|---:|---:|---|
| bm25-prompt | 0.402027 | 276/276 | 1651.6 | 0.642 s | Retrieved booklet passages |
| bm25-score | 0.436482 | 276/276 | 1654.6 | 0.401 s | Retrieved booklet passages |
| dense-prompt | 0.509928 (valid subset) | 274/276 | 1396.1 | 0.601 s | Retrieved booklet passages |
| dense-score | 0.498261 | 276/276 | 1395.8 | 0.359 s | Retrieved booklet passages |
| full-prompt | 0.552448 | 276/276 | 14103.1 | 3.911 s | Full/capped source context |
| full-score | 0.671612 | 276/276 | 14106.1 | 3.635 s | Full/capped source context |
| hybrid-prompt | 0.496620 | 276/276 | 1653.6 | 0.667 s | Retrieved booklet passages |
| hybrid-score | 0.526883 | 276/276 | 1656.6 | 0.426 s | Retrieved booklet passages |
| reference-prompt | 0.531363 | 276/276 | 2266.7 | 0.779 s | Oracle/reference only |
| reference-score | 0.535419 | 276/276 | 2269.7 | 0.540 s | Oracle/reference only |
| full / option_logits | 0.721862 | 276/276 | 14106.1 | 3.636 s | Full/capped source context |
| full / hidden | 0.894461 | 276/276 | 14106.1 | 3.636 s | Full/capped source context |

Dense generation has two invalid decisions; its score uses 274 valid outputs and the configuration is excluded from selection. All other validation setups have complete outputs. Oracle evidence is a diagnostic, not a strict upper bound: larger context can help. All production modes use booklet + claim only. Full/capped context was truncated for all 276 validation rows at the 48,000-byte document-context budget. Head validation latency is cached A40 encoding plus separately measured CPU classifier time; it is not a live integrated service measurement.

The selected hidden head uses 4096 features. StandardScaler and multinomial logistic regression were fitted within each training-only connected-group fold. The C grid is 0.001, 0.01, 0.1, 1; selected C is 0.01, and training-only OOF temperature is 2.148798. Training has only two connected groups (fold sizes 838, 64), limiting confidence. The predeclared gate was +0.02 validation Macro-F1 over the best booklet-only baseline; observed gain 0.222849 passed. Test labels were excluded from fitting, calibration and selection.

**Frozen internal holdout: 310 examples, Macro-F1 0.899920, accuracy 0.900000, average context 14070.0 tokens, average actual inference plus measured retrieval 3.281 seconds on RTX A6000.** Startup, weight verification/download, PDF extraction and interrupted downtime are excluded from per-example latency. The additional classifier requires no additional Apertus forward pass; its training and lease time are still costs. 310/310 heldout inputs report context truncation.

| Official class | Holdout F1 | Support |
|---|---:|---:|
| 0 Entailment | 0.870000 | 92 |
| 1 Neutral | 0.956522 | 104 |
| 2 Contradiction | 0.873239 | 114 |

| Claim language | Validation Macro-F1 | Holdout Macro-F1 | Holdout n |
|---|---:|---:|---:|
| DE | 0.878468 | 0.881537 | 109 |
| FR | 0.886981 | 0.889234 | 86 |
| IT | 0.911735 | 0.922059 | 115 |

Cross-language holdout: 205 examples, Macro-F1 0.889426.

| Document → claim | Validation Macro-F1 | Holdout Macro-F1 | Holdout n |
|---|---:|---:|---:|
| DE → DE | 0.915344 | 0.843640 | 26 |
| FR → DE | 0.835940 | 0.885931 | 53 |
| IT → DE | 0.911111 | 0.859025 | 30 |
| DE → FR | 0.710317 | 0.866667 | 23 |
| FR → FR | 0.908751 | 0.851010 | 34 |
| IT → FR | 0.973374 | 0.958170 | 29 |
| DE → IT | 0.790065 | 0.911111 | 35 |
| FR → IT | 0.938841 | 0.845192 | 35 |
| IT → IT | 1.000000 | 1.000000 | 45 |

Holdout calibration: ECE 0.083631, Brier 0.168119, NLL 0.315898; temperature was fixed using train-only OOF scores. Evidence-ID metrics contain no annotated full-booklet gold spans and must not be treated as evidence-quality scores. Actual PDF CLI evidence quotes/pages were checked for provenance; minimality, relevance and causal model attribution are not established.

Legacy timings are offline GPU encoding plus separately timed CPU head, not live CPU
or integrated service latency. Legacy exact-deduplicated score is 0.965299 on 207 rows.
Training head/calibration used train only. These old results must not be renamed
v1.5 or full-booklet scores. Historical details/negative results are preserved in
`docs/legacy_apertus_2509_report.md` and experiment directories.

## 6. Limitations

The 310 examples are an internal heldout split, not organizer-hidden evaluation. Only two connected groups in train and validation, duplicate/translated examples and small language slices limit generalization claims; no confidence interval based on independent documents is asserted. No matched base-model holdout run was made. Full context is capped at 48,000 document bytes, so later evidence can be omitted. Retrieved/supplied passages are source-grounded candidates, not validated minimal proof or model attribution. Semantic evidence quality has not been manually scored; reference overlap is approximate. No OCR, LoRA, weight fine-tuning, or official hidden evaluation was performed. Native v1.5 needs the pinned fork and considerable RAM/VRAM; CPU operation is slow. The default Docker frontend requires a separately configured compatible backend. No unspecified organizer interface, hardware or network restrictions are invented.

## 7. Reproducibility

Clean-checkout `make run` builds the lightweight PDF/HTTP Dockerfile, installs
hash-locked dependencies and starts port 8000. The selected full-context frontend
requires no supporting model download. Optional `RUNTIME=hybrid` supports E5 retrieval. Classification needs
an authorized v1.5 endpoint; unavailable inference produces an explicit failure.
Project-defined CLI: `python -m ost_nli predict BOOKLET.pdf CLAIM`; its default
context comes from `deployment/v15-selection.json`. Explicit overrides are documented.
Batch JSONL contains id/document/claim only. Source passages carry pages/offsets/hash.
Documented interface is separable from model/retrieval implementation.

The recorded host and freshly rebuilt Docker image pass 38 tests each; the restored current host also passes 38 tests and make run processed a real 40-page PDF with verified source quotes/pages (experiments/v15-restored-frontend-v1); `experiments/v15-production-context-v1` verifies full/capped source handling on a real French PDF and includes an HTTP regression that sends the same full context through CLI and web. The earlier 37-test image is recorded separately;
`experiments/v15-fresh-docker-v1` records a real PDF upload and hybrid evidence
retrieval through `make run` without mounting host source into the image. The
following earlier clean-checkout check ran 36 tests. A clean GitHub checkout successfully
ran `make run`, downloaded fresh public E5 weights, and retrieved source passages
from an official French PDF for German, French and Italian claims. Source hashes,
pages and quote offsets were checked. Unconfigured v1.5 prediction correctly fails
without returning a label. Records are in `experiments/contract-software-v1/`;
the separate synthetic HTTP batch fixture does not constitute real v1.5 inference.
The final deployment image also passed all 38 tests with no failures/skips; /benchmarks serves the exact final JSON and the packaged head SHA matches the frozen source (experiments/v15-final-frontend-v1). Its current frontend has no persistent NLI backend configured; actual selected v1.5 NLI was verified by the three real GPU PDF CLI cases. CPU PDF/retrieval checks are separate from GPU benchmark performance.
Lease costs, failures, active IDs, cleanup deadlines and verified destruction are
recorded in `experiments/budget.json`. Provisioning failures produce no NLI score.
The USD 10 hard budget and USD 2 reserve remain binding; billing is asynchronous.

All 21 recorded task leases were destroyed and verified absent from the paginated provider listing. Observed account-credit reduction is USD 3.465574; observed credit remaining is USD 6.534426, checked at 2026-10-07T20:45:40.024679+00:00. No outstanding task lease remains. Billing may settle asynchronously. The temporary task SSH key was removed and unrelated keys preserved; task HF secrets remain absent from Vast. The new cloud HF_TOKEN binding was retained and its gated configuration access verified without downloading weights. See experiments/v15-final-provider-cleanup.json and budget.json.


Reproduce the selected backend with verified weights and the pinned native dependencies:

```bash
PYTHONPATH=track_2a/src python track_2a/scripts/serve_v15.py \
  --model-dir /path/to/verified/apertus-v15-model --device cuda \
  --method head --head track_2a/deployment/head-v15.json --port 8001
```

On Python 3.11/Linux install requirements-v15-cuda.lock, requirements-v15.lock, then requirements-head-v15.lock to match the native head runtime; run scripts/check_v15_cuda.py. The frontend needs LLM_NAME=swiss-ai/Apertus-v1.5-8B and a reachable LLM_BASE_URL ending /v1 for this head-enabled service; ordinary base-model chat endpoints do not reproduce the reported classifier. The server above listens on loopback. On Linux a Docker frontend can use --network host and LLM_BASE_URL=http://127.0.0.1:8001/v1; configure NO_PROXY for local requests. Model weights and adapters are not downloaded into the repository. Download authorized weights once through ost_nli.v15.download(directory, os.environ["HF_TOKEN"]) in the native environment; it pins the revision and checks every original shard hash. Then use the offline cache. Generic .venv does not establish native-fork readiness.

Scientific raw artifacts: experiments/apertus-v15-research-v1 (validation, training, both heads, final selection/predictions/metrics, model/input manifests, resume audit and actual CLI output). Large binary features remain ignored; model/data downloads are reproducible separately. The original inference/frozen-selection commit is 15c6e57b4992a513ffa5f5eec350282e11dd63d5. Resume/PDF-path orchestration was committed as 9910fe36ad1a90bbd182737a2a4ff8579aaf42c6, copied into that checkout, and separately SHA verified. The engine and frozen head/inputs were unchanged; the dirty runtime tree is explicitly recorded. Archive SHA, every source file SHA, split fingerprints, initial 26 predictions and deployed head SHA were verified locally; Macro-F1 was independently recomputed from the confusion matrix (experiments/v15-delivery-integrity.json).

## 8. Next steps

Research and the internal frozen holdout are complete. Further work should use new development documents for improving evidence relevance and context compression; do not tune against the 310 holdout. Confirm any subsequently published evaluator adapter and arrange a documented v1.5 backend for judging. Actual submission through the organizer site is a separate action.

## 9. License

Code Apache-2.0; report CC-BY-4.0 following the template. OST dataset metadata MIT;
multilingual E5 MIT; Apertus v1.5 Apache-2.0 with published access/AUP conditions;
pypdf BSD-3-Clause. Original booklet content remains attributed to the Federal Chancellery.

## 10. References

- Adopted user-provided OST contract: `docs/challenge_contract.md`.
- https://github.com/HackApertus/project-template/tree/main/track_2a
- https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets
- https://www.bk.admin.ch/de/sammlung-der-abstimmungsbuechlein-seit-1978
- https://huggingface.co/swiss-ai/Apertus-v1.5-8B
- https://huggingface.co/intfloat/multilingual-e5-small
- Submission: http://hackapertus.ch/online-hack/submissions


## Phase 2: grounding controls passed; efficiency study registered

The historical310 score is consumed and frozen; it is not an architecture-development set. Phase2 registered grounding controls before new GPU results, with a USD4.50 discretionary ceiling and USD2 reserve. LoRA/QLoRA and70B are prohibited in this stage. Current production selection remains unchanged pending grounding controls.

Local train-only CPU probes achieve validation Macro-F1 .620765 (wordTFIDF), .626285 (characterTFIDF), and .507226 (language/length metadata). These are supplementary artifact controls, not an Apertus claim-only experiment.

The original full-trained hidden head on original cached validation encodings yields .861704 with hybridk5 (~1657 prompt tokens), .848070 with densek5 (~1396), .661108 with BM25k5 (~1655), and .940951 with oracle reference (~2270). Same full head/weights, no refitting or new GPU calls: these context interventions diagnose sensitivity, not a context-matched head sweep or a final Pareto-selected architecture. Hybrid is a promising efficiency candidate; optimization waits for source-dependence controls.

Validation contains276 rows but207 unique normalized claim/document pairs, three voting events and two duplicate-connected components. Deduplicated Macro-F1 is .883035. An inspected Neutral example, ost-v1.1-0732, has a COVID reference while full booklet pages6/32 discuss the claim’s climate transition/2050 target. This exposes a possible reference-relative label versus full-booklet premise-scope mismatch; labels are unchanged and the counterexample is preserved with exact page text and PDF SHA. It prevents treating every scored error as a reasoning failure.

See [phase2 audit](experiments/apertus-v15-phase2/README.md), [registered protocol](experiments/apertus-v15-phase2/protocol.json), [scope counterexample](experiments/apertus-v15-phase2/premise-scope-risk.json), and [validation error inventory](experiments/apertus-v15-phase2/final-error-analysis.jsonl). Actual native controls completed: correct full .894461, claim-only condition-specific train-only fitted head .699640, original full head on empty .210229, wrong-event same-language booklets .565341/.553328 (seeds42/1337), generic .174688. The preregistered retention/absolute-gap gate passes with limitations. Large wrong-context drops support context dependence, while .700 claim-only capability exposes claim-label signal. Original labels are sensitivity targets, not swapped-premise NLI ground truth. All2006 new forwards used train/validation only; all export/source checksums passed, and the first GPU lease was destroyed and verified absent. No LoRA and no new310 inference. Context-matched heads at1k/2k/4k/8k hybrid budgets and a2k prefix contrast are registered before those results; joint depth/cap changes are not a factorial study.
