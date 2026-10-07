# Apertus Evidence Lab — Track 2A OST technical report

## 1. Summary

The adopted challenge contract requires booklet + claim, fixed labels 0 entailment /
1 neutral / 2 contradiction, Apertus v1.5, and transparent source passages. Production
must not require reference strings. The app now implements PDF processing,
multilingual semantic/BM25 fusion retrieval, source provenance, a Docker UI and
single/batch CLI. Exact CLI/evidence schemas are project choices, not blockers.

**All ten real native v1.5 validation comparisons are complete.** The best baseline is full/capped booklet context with restricted class scoring: preliminary Macro-F1 **0.671612** on all 276 validation rows, with zero invalid outputs. Two decision heads will be fitted using only the 902 training rows; a head must improve validation Macro-F1 by at least 0.02 before deployment. The single 310-row internal holdout run is still pending. These log-reported values await SHA-verified artifact recovery.

The runtime uses an A40, original SHA-verified v1.5 weights, Torch 2.8.0/CUDA 12.8 and the pinned native fork. The temporary Hugging Face credential was removed from Vast and can be revoked. Native CPU/public PDF CLI integration and Docker checks are recorded separately; they are not benchmark estimates. The previous 0.970353 result used Apertus 2509 with supplied references and remains historical research.

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
implemented. Full context is explicitly byte-capped/truncation-reported when used.

## 3. Use of Apertus

Required model: `swiss-ai/Apertus-v1.5-8B`, pinned metadata revision
`a411d838600baf0e3635a3daf66fb7c55fc97bb6`, architecture
`Apertus1p5ForConditionalGeneration`, model type `apertus1p5`. It is distinct from
legacy Apertus 2509; old hidden-head parameters cannot be transferred unchanged.
The current production interface uses an authorized v1.5 endpoint through
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
0.116697/0.176814/0.203536. No holdout NLI Macro-F1 has been measured.

This diagnostic measures overlap with available references, not semantic evidence
correctness, official evidence score, or NLI Macro-F1. Dense slightly exceeds hybrid
on cross-language overlap; hybrid wins aggregate overlap. NLI superiority is unproven.

| Validation setup | Macro-F1 | Valid outputs | Context tokens | Latency | Evidence |
|---|---:|---:|---:|---:|---|
| reference / score | 0.535419 (preliminary) | 276/276 | Recovery pending | Recovery pending | Oracle/reference diagnostic only |
| reference / prompt | 0.531363 (preliminary) | 276/276 | Recovery pending | Recovery pending | Oracle/reference diagnostic only |
| bm25 / score | 0.436482 (preliminary) | 276/276 | Recovery pending | Recovery pending | Real booklet retrieval |
| bm25 / prompt | 0.402027 (preliminary) | 276/276 | Recovery pending | Recovery pending | Real booklet retrieval |
| dense / score | 0.498261 (preliminary) | 276/276 | Recovery pending | Recovery pending | Real booklet retrieval |
| dense / prompt | 0.509928 (valid subset) (preliminary) | 274/276 | Recovery pending | Recovery pending | Real booklet retrieval |
| hybrid / score | 0.526883 (preliminary) | 276/276 | Recovery pending | Recovery pending | Real booklet retrieval |
| hybrid / prompt | 0.496620 (preliminary) | 276/276 | Recovery pending | Recovery pending | Real booklet retrieval |
| full / score | 0.671612 (preliminary) | 276/276 | Recovery pending | Recovery pending | Full/capped booklet input |
| full / prompt | 0.552448 (preliminary) | 276/276 | Recovery pending | Recovery pending | Full/capped booklet input |

Dense generation produced two invalid JSON decisions. Its Macro-F1 is on the 274 valid outputs; configurations with any invalid output are excluded from final selection. Oracle/reference results are diagnostic and are not a strict upper bound: additional booklet context can improve inference. All production comparisons use booklet + claim only.

Legacy timings are offline GPU encoding plus separately timed CPU head, not live CPU
or integrated service latency. Legacy exact-deduplicated score is 0.965299 on 207 rows.
Training head/calibration used train only. These old results must not be renamed
v1.5 or full-booklet scores. Historical details/negative results are preserved in
`docs/legacy_apertus_2509_report.md` and experiment directories.

## 6. Limitations

Full-booklet production NLI readiness and final F1 remain unverified until the
real GPU benchmark finishes. Public PDF CLI/native CPU integration was verified
on three convenience-selected training examples, but this does not establish F1. Authorized v1.5 access and
native CPU execution have been verified. The source reference is not always a uniquely
aligned minimal passage, and approximate overlap loses typography/word-order matches.
Retrieved passages are transparent candidates, not proof of model attribution or
correct evidence. Small independent group counts, duplicated examples, translated
near-duplicates and reference-to-whole-booklet distribution shift limit conclusions.
No OCR, dense-model fine-tuning, v1.5 head training/LoRA or official hidden evaluation
has been performed. No unspecified interface/hardware limits are invented.

## 7. Reproducibility

Clean-checkout `make run` builds the lightweight PDF/HTTP Dockerfile, installs
hash-locked dependencies and starts port 8000. The selected full-context frontend
requires no supporting model download. Optional `RUNTIME=hybrid` supports E5 retrieval. Classification needs
an authorized v1.5 endpoint; unavailable inference produces an explicit failure.
Project-defined CLI: `python -m ost_nli predict BOOKLET.pdf CLAIM`; its default
context comes from `deployment/v15-selection.json`. Explicit overrides are documented.
Batch JSONL contains id/document/claim only. Source passages carry pages/offsets/hash.
Documented interface is separable from model/retrieval implementation.

The latest host and freshly rebuilt Docker image pass 38 tests each; `experiments/v15-production-context-v1` verifies full/capped source handling on a real French PDF and includes an HTTP regression that sends the same full context through CLI and web. The earlier 37-test image is recorded separately;
`experiments/v15-fresh-docker-v1` records a real PDF upload and hybrid evidence
retrieval through `make run` without mounting host source into the image. The
following earlier clean-checkout check ran 36 tests. A clean GitHub checkout successfully
ran `make run`, downloaded fresh public E5 weights, and retrieved source passages
from an official French PDF for German, French and Italian claims. Source hashes,
pages and quote offsets were checked. Unconfigured v1.5 prediction correctly fails
without returning a label. Records are in `experiments/contract-software-v1/`;
the separate synthetic HTTP batch fixture does not constitute real v1.5 inference.
CPU PDF/retrieval checks are separate from the real v1.5 GPU research attempts.
Lease costs, failures, active IDs, cleanup deadlines and verified destruction are
recorded in `experiments/budget.json`. Provisioning failures produce no NLI score.
The USD 10 hard budget and USD 2 reserve remain binding; billing is asynchronous.

## 8. Next steps

All ten matched native v1.5 validation comparisons are complete. Full/capped class
scoring leads the booklet-only baselines. Finish extracting the 902 training features
and fitting the two decision heads on train only,
freeze choices, then evaluate the 310 internal holdout once for NLI. Measure evidence
quality manually alongside reference-overlap diagnostics. No Devpost access or exact
organizer CLI signature is needed. Submission itself remains a separate action.

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
